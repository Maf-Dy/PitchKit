package com.nicos.pitchkit.tuner

import android.media.MediaRecorder
import android.util.Log
import com.nicos.pitchkit.tuner.harmony.ChordRecognizer
import com.nicos.pitchkit.tuner.models.AudioFrame
import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlin.math.sqrt

internal class TunerEngine(
    profile: InstrumentProfile = InstrumentProfile.Guitar,
    private val mode: DetectionMode = DetectionMode.AUTO,
    referenceA4Hz: Double = 440.0,
    highPassCutoffHz: Double = 30.0,
    autoChordThreshold: Double = 0.30,
    chordMinScore: Double = 0.20,
    private val chordRecognizer: ChordRecognizer? = null,
    preferredSampleRate: Int = 44100,
    bufferSize: Int = 8192,
    preferredAudioSource: Int = MediaRecorder.AudioSource.MIC,
) : AutoCloseable {
    private companion object {
        const val CHORD_GATE_ATTACK_FRAMES = 2
        const val CHORD_GATE_RESET_FRAMES = 4
        const val CHORD_GATE_OPEN_MULTIPLIER = 1.15
        const val CHORD_GATE_HOLD_MULTIPLIER = 0.75

        /** Drops are expected under load; one line per this many keeps logcat readable. */
        const val DROP_LOG_INTERVAL = 50L
    }

    private val capture = AudioCapture(
        preferredSampleRate = preferredSampleRate,
        bufferSize = bufferSize,
        preferredAudioSource = preferredAudioSource,
    )
    private val analyzer = PitchAnalyzer(
        profile = profile,
        mode = mode,
        referenceA4Hz = referenceA4Hz,
        highPassCutoffHz = highPassCutoffHz,
        autoChordThreshold = autoChordThreshold,
        chordMinScore = chordMinScore,
    )
    private val chordRmsGate = profile.rmsGate.coerceAtLeast(0.0)

    /**
     * Counts what the conflated capture channel discards. Exposed on the emitted
     * [TuningResult.Chord] so the live screen can show it behind a debug flag;
     * the channel itself is deliberately unchanged.
     */
    private val frameDrops = FrameDropCounter()

    @Volatile
    private var closed = false

    private var chordGateOpen = false
    private var chordAttackFrames = 0
    private var chordQuietFrames = 0

    fun start() = callbackFlow<TuningResult> {
        // The UI can race a recomposition-driven collection against the owner's
        // close(); complete the flow instead of crashing the collector.
        if (closed) {
            close()
            awaitClose { }
            return@callbackFlow
        }

        frameDrops.reset()
        // Teardown hands every buffered frame to onUndeliveredElement too; that is
        // shutdown, not conflation, so it must not be counted as a drop.
        var draining = false

        val frames = Channel<AudioFrame>(
            capacity = Channel.CONFLATED,
            onUndeliveredElement = { frame ->
                if (!draining && frameDrops.onDropped() % DROP_LOG_INTERVAL == 0L) {
                    Log.w("PitchKitAudio", "conflated capture: ${frameDrops.snapshot().describe()}")
                }
                capture.recycle(frame.samples)
            },
        )

        val processor = launch(Dispatchers.Default) {
            for (frame in frames) {
                try {
                    frameDrops.onProcessed()
                    if (closed) break
                    val result = try {
                        if (mode == DetectionMode.CHORD && chordRecognizer != null) {
                            processNeuralChordFrame(frame)
                        } else {
                            analyzer.process(frame)
                        }
                    } catch (error: IllegalStateException) {
                        // The recognizer's ONNX session can be closed by its owner
                        // while a frame is mid-flight; stop processing quietly.
                        break
                    }
                    this@callbackFlow.trySend(withDiagnostics(result))
                } finally {
                    capture.recycle(frame.samples)
                }
            }
        }

        capture.start(scope = this) { samples, actualSampleRate ->
            if (closed) {
                capture.recycle(samples)
            } else {
                val sendResult = frames.trySend(
                    AudioFrame(
                        samples = samples,
                        sampleRate = actualSampleRate,
                        channelCount = 1,
                    )
                )
                if (sendResult.isFailure) capture.recycle(samples) else frameDrops.onDelivered()
            }
        }

        awaitClose {
            draining = true
            frames.cancel()
            processor.cancel()
            stop()
        }
    }
        .buffer(Channel.CONFLATED)
        .flowOn(Dispatchers.IO)

    /**
     * Attach what only the engine knows: which audio source the negotiation
     * actually opened, and how many capture buffers the conflated channel has
     * thrown away so far. Both are diagnostics — nothing downstream decides on
     * them — so they ride on the chord result rather than changing any control flow.
     */
    private fun withDiagnostics(result: TuningResult): TuningResult = when (result) {
        is TuningResult.Chord -> result.copy(
            backend = capture.activeSourceLabel
                ?.let { source -> "${result.backend} · $source" }
                ?: result.backend,
            droppedFrames = frameDrops.droppedCount(),
        )
        else -> result
    }

    private fun processNeuralChordFrame(frame: AudioFrame): TuningResult {
        val rms = frameRms(frame)
        val openThreshold = chordRmsGate * CHORD_GATE_OPEN_MULTIPLIER
        val holdThreshold = chordRmsGate * CHORD_GATE_HOLD_MULTIPLIER

        if (!chordGateOpen) {
            if (rms >= openThreshold) {
                chordAttackFrames++
            } else {
                chordAttackFrames = 0
            }

            if (chordAttackFrames < CHORD_GATE_ATTACK_FRAMES) {
                return TuningResult.Silence
            }

            chordGateOpen = true
            chordAttackFrames = 0
            chordQuietFrames = 0
        } else if (rms < holdThreshold) {
            chordQuietFrames++
            if (chordQuietFrames >= CHORD_GATE_RESET_FRAMES) {
                resetChordGate()
                chordRecognizer?.reset()
            }
            return TuningResult.Silence
        } else {
            chordQuietFrames = 0
        }

        return chordRecognizer?.recognize(frame)
            ?.let {
                TuningResult.Chord(
                    name = it.label,
                    confidence = it.confidence,
                    backend = it.backend,
                )
            }
            ?: TuningResult.Silence
    }

    private fun frameRms(frame: AudioFrame): Double {
        val samples = frame.samples
        if (samples.isEmpty()) return 0.0

        var energy = 0.0
        for (sample in samples) {
            val value = sample.toDouble()
            energy += value * value
        }
        return sqrt(energy / samples.size)
    }

    private fun resetChordGate() {
        chordGateOpen = false
        chordAttackFrames = 0
        chordQuietFrames = 0
    }

    fun stop() {
        if (closed) return
        capture.stop()
        analyzer.reset()
        chordRecognizer?.reset()
        resetChordGate()
    }

    @Synchronized
    override fun close() {
        if (closed) return
        capture.stop()
        analyzer.reset()
        chordRecognizer?.reset()
        resetChordGate()
        closed = true
    }
}
