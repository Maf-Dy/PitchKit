package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.AudioFrame
import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn

/**
 * Android microphone convenience engine. DSP lives in [PitchAnalyzer]; this class
 * only turns microphone samples into [AudioFrame] objects.
 */
internal class TunerEngine(
    profile: InstrumentProfile = InstrumentProfile.Guitar,
    mode: DetectionMode = DetectionMode.AUTO,
    referenceA4Hz: Double = 440.0,
    highPassCutoffHz: Double = 30.0,
    autoChordThreshold: Double = 0.30,
    chordMinScore: Double = 0.20,
    private val sampleRate: Int = 44100,
    bufferSize: Int = 8192,
) {
    private val capture = AudioCapture(sampleRate, bufferSize)
    private val analyzer = PitchAnalyzer(
        profile = profile,
        mode = mode,
        referenceA4Hz = referenceA4Hz,
        highPassCutoffHz = highPassCutoffHz,
        autoChordThreshold = autoChordThreshold,
        chordMinScore = chordMinScore,
    )

    fun start() = callbackFlow<TuningResult> {
        capture.start(scope = this) { samples ->
            trySend(
                analyzer.process(
                    AudioFrame(
                        samples = samples,
                        sampleRate = sampleRate,
                        channelCount = 1,
                    )
                )
            )
        }
        awaitClose { stop() }
    }
        .buffer(capacity = Channel.CONFLATED)
        .flowOn(Dispatchers.IO)

    fun stop() {
        capture.stop()
        analyzer.reset()
    }
}
