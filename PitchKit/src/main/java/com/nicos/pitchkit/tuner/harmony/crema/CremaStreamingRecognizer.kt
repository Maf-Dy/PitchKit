package com.nicos.pitchkit.tuner.harmony.crema

import android.os.SystemClock
import android.util.Log
import com.nicos.pitchkit.BuildConfig
import com.nicos.pitchkit.tuner.harmony.ChordRecognition
import com.nicos.pitchkit.tuner.harmony.ChordRecognizer
import com.nicos.pitchkit.tuner.harmony.ChordStabilizer
import com.nicos.pitchkit.tuner.harmony.chordnet.FloatRingBuffer
import com.nicos.pitchkit.tuner.harmony.chordnet.StreamingPcmResampler
import com.nicos.pitchkit.tuner.models.AudioFrame

/** Low-latency rolling Crema 0.2.0 recognizer for live instrument harmony. */
class CremaStreamingRecognizer(
    modelBytes: ByteArray,
    runtimeStateJson: String,
    harmonic1PlanBytes: ByteArray,
    harmonic2PlanBytes: ByteArray,
    referenceA4Hz: Double = 440.0,
    preferFlats: Boolean = false,
    minimumConfidence: Double = 0.08,
) : ChordRecognizer {
    private companion object {
        const val LIVE_CONTEXT_FRAMES = 24
        const val STARTUP_FRAMES = 8
        const val INFERENCE_STRIDE_FRAMES = 2
    }

    private val frontend = CremaHcqtFrontend(harmonic1PlanBytes, harmonic2PlanBytes)
    private val runner = CremaOnnxRunner(modelBytes)
    private val decoder = CremaHarmonyDecoder(
        state = CremaRuntimeState.parse(runtimeStateJson),
        preferFlats = preferFlats,
    )
    private val resampler = StreamingPcmResampler(
        targetRate = CremaContract.SAMPLE_RATE,
        pitchScale = 440.0 / referenceA4Hz,
    )
    private val audio = FloatRingBuffer(LIVE_CONTEXT_FRAMES * CremaContract.HOP_LENGTH)
    private val stabilizer = ChordStabilizer(
        minimumConfidence = minimumConfidence,
        changeConfirmations = 2,
    )

    private var totalTargetSamples = 0L
    private var lastInferenceAt = 0L
    private var closed = false
    private var inferenceCount = 0

    init {
        require(referenceA4Hz in 300.0..600.0) {
            "referenceA4Hz must be between 300 and 600 Hz"
        }
    }

    @Synchronized
    override fun recognize(frame: AudioFrame): ChordRecognition? {
        if (closed) return null

        val targetSamples = resampler.process(frame.toMono(), frame.sampleRate)
        if (targetSamples.isEmpty()) return stabilizer.currentWithoutPrediction()

        audio.append(targetSamples)
        totalTargetSamples += targetSamples.size

        val minimumSamples = STARTUP_FRAMES * CremaContract.HOP_LENGTH
        if (audio.size < minimumSamples) return stabilizer.currentWithoutPrediction()

        val inferenceStride = INFERENCE_STRIDE_FRAMES * CremaContract.HOP_LENGTH
        if (totalTargetSamples - lastInferenceAt < inferenceStride) {
            return stabilizer.currentWithoutPrediction()
        }
        lastInferenceAt = totalTargetSamples

        val startedAt = SystemClock.elapsedRealtimeNanos()
        val features = frontend.transform(audio.toFloatArray())
        val hcqtDoneAt = SystemClock.elapsedRealtimeNanos()
        if (features.frameCount <= 0 || closed) return stabilizer.update(null)

        val heads = runner.infer(features.values, features.frameCount)
        val onnxDoneAt = SystemClock.elapsedRealtimeNanos()
        val prediction = decoder.decode(heads)

        logTiming(startedAt, hcqtDoneAt, onnxDoneAt, features.frameCount)

        val rawRecognition = prediction?.let {
            ChordRecognition(
                label = it.label,
                confidence = it.confidence,
                backend = "Crema 0.2.0",
            )
        }
        val emitted = stabilizer.update(rawRecognition)
        logDecision(
            prediction = prediction,
            emitted = emitted,
        )
        return emitted
    }

    @Synchronized
    override fun reset() {
        if (closed) return
        resetState()
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        frontend.close()
        runner.close()
        resetState()
    }

    private fun resetState() {
        audio.clear()
        resampler.reset()
        totalTargetSamples = 0L
        lastInferenceAt = 0L
        stabilizer.reset()
        inferenceCount = 0
    }

    private fun logDecision(
        prediction: CremaDecodedChord?,
        emitted: ChordRecognition?,
    ) {
        if (!BuildConfig.DEBUG) return
        val top = prediction?.alternatives
            ?.joinToString(separator = " | ") { candidate ->
                "${candidate.label}=fit:${"%.3f".format(candidate.fit)},tag:${"%.3f".format(candidate.tagConfidence)}"
            }
            .orEmpty()
        Log.d(
            "PitchKitChord",
            "backend=Crema raw=${prediction?.label ?: "N"} model=${prediction?.rawLabel ?: "N"} " +
                "conf=${prediction?.confidence?.let { "%.3f".format(it) } ?: "-"} " +
                "top3=[$top] emitted=${emitted?.label ?: "-"}",
        )
    }

    private fun logTiming(
        startedAt: Long,
        hcqtDoneAt: Long,
        onnxDoneAt: Long,
        frames: Int,
    ) {
        if (!BuildConfig.DEBUG) return
        inferenceCount++
        if (inferenceCount % 8 != 0) return

        val hcqtMs = (hcqtDoneAt - startedAt) / 1_000_000.0
        val onnxMs = (onnxDoneAt - hcqtDoneAt) / 1_000_000.0
        val totalMs = (onnxDoneAt - startedAt) / 1_000_000.0
        Log.d(
            "PitchKitPerf",
            "Crema frames=$frames hcqt=${"%.1f".format(hcqtMs)}ms " +
                "onnx=${"%.1f".format(onnxMs)}ms total=${"%.1f".format(totalMs)}ms",
        )
    }
}
