package com.nicos.pitchkit.tuner.harmony.chordnet

import android.util.Log
import com.nicos.pitchkit.BuildConfig
import com.nicos.pitchkit.tuner.harmony.ChordRecognition
import com.nicos.pitchkit.tuner.harmony.ChordRecognizer
import com.nicos.pitchkit.tuner.harmony.ChordStabilizer
import com.nicos.pitchkit.tuner.models.AudioFrame
import java.security.MessageDigest

/** Low-latency rolling recognizer for ChordNet 2E1D. */
class ChordNetStreamingRecognizer(
    modelBytes: ByteArray,
    planBytes: ByteArray,
    referenceA4Hz: Double = 440.0,
    minimumConfidence: Double = 0.08,
) : ChordRecognizer {
    private val plan: CqtPlan
    private val frontend: CqtCpuFrontend
    private val runner: ChordNetOnnxRunner
    private val resampler: StreamingPcmResampler
    private val stabilizer = ChordStabilizer(
        minimumConfidence = minimumConfidence,
        changeConfirmations = 2,
    )

    private companion object {
        const val LIVE_CONTEXT_FRAMES = 32
        const val STARTUP_FRAMES = 10
        const val INFERENCE_STRIDE_FRAMES = 2
        const val LIVE_SMOOTHING_KERNEL = 5
    }

    private val maxSamples = LIVE_CONTEXT_FRAMES * ChordNetContract.HOP_LENGTH - 1
    private val audio = FloatRingBuffer(maxSamples)

    private var totalTargetSamples = 0L
    private var lastInferenceAt = 0L

    init {
        require(referenceA4Hz in 300.0..600.0) { "referenceA4Hz must be between 300 and 600 Hz" }
        require(sha256(modelBytes) == ChordNetContract.MODEL_SHA256) {
            "Unexpected ChordNet model SHA-256"
        }
        require(sha256(planBytes) == ChordNetContract.PLAN_SHA256) {
            "Unexpected ChordNet CQT plan SHA-256"
        }
        plan = CqtPlanDecoder.decodeAndVerify(planBytes)
        frontend = CqtCpuFrontend(plan)
        runner = ChordNetOnnxRunner(modelBytes)
        resampler = StreamingPcmResampler(
            pitchScale = 440.0 / referenceA4Hz,
        )
    }

    override fun recognize(frame: AudioFrame): ChordRecognition? {
        val targetSamples = resampler.process(frame.toMono(), frame.sampleRate)
        if (targetSamples.isEmpty()) return stabilizer.currentWithoutPrediction()

        audio.append(targetSamples)
        totalTargetSamples += targetSamples.size

        val minimumSamples = (STARTUP_FRAMES - 1) * ChordNetContract.HOP_LENGTH
        if (audio.size < minimumSamples) return stabilizer.currentWithoutPrediction()

        val inferenceStride = INFERENCE_STRIDE_FRAMES * ChordNetContract.HOP_LENGTH
        if (totalTargetSamples - lastInferenceAt < inferenceStride) {
            return stabilizer.currentWithoutPrediction()
        }
        lastInferenceAt = totalTargetSamples

        val features = frontend.transform(audio.toFloatArray())
        if (features.frameCount <= 0) return stabilizer.update(null)

        val validFrames = features.frameCount.coerceAtMost(ChordNetContract.SEQUENCE_LENGTH)
        val modelInput = FloatArray(
            ChordNetContract.SEQUENCE_LENGTH * ChordNetContract.INPUT_BINS
        )

        if (features.frameCount <= ChordNetContract.SEQUENCE_LENGTH) {
            features.values.copyInto(
                destination = modelInput,
                endIndex = validFrames * ChordNetContract.INPUT_BINS,
            )
        } else {
            val firstFrame = features.frameCount - ChordNetContract.SEQUENCE_LENGTH
            val sourceStart = firstFrame * ChordNetContract.INPUT_BINS
            features.values.copyInto(
                destination = modelInput,
                startIndex = sourceStart,
                endIndex = sourceStart + modelInput.size,
            )
        }

        val logits = runner.infer(modelInput, windowCount = 1)
        val predictions = ChordNetPostProcessor.decode(
            logits = logits,
            windowCount = 1,
            validFrameCount = validFrames,
            smoothingKernel = LIVE_SMOOTHING_KERNEL,
            includeAlternatives = BuildConfig.DEBUG,
        )

        val prediction = predictions[validFrames - 1]
        val rawRecognition = prediction.displayLabel?.let {
            ChordRecognition(
                label = it,
                confidence = prediction.confidence,
                backend = "ChordNet 2E1D",
            )
        }
        val emitted = stabilizer.update(rawRecognition)
        if (BuildConfig.DEBUG) {
            val top = prediction.alternatives.joinToString(separator = " | ") { candidate ->
                "${candidate.displayLabel ?: candidate.rawLabel}=${"%.3f".format(candidate.confidence)}"
            }
            Log.d(
                "PitchKitChord",
                "backend=ChordNet raw=${prediction.displayLabel ?: "N"} " +
                    "model=${prediction.rawLabel} conf=${"%.3f".format(prediction.confidence)} " +
                    "top3=[$top] emitted=${emitted?.label ?: "-"}",
            )
        }
        return emitted
    }

    override fun reset() {
        audio.clear()
        resampler.reset()
        totalTargetSamples = 0L
        lastInferenceAt = 0L
        stabilizer.reset()
    }

    override fun close() {
        runner.close()
        reset()
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
