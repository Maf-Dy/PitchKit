package com.nicos.pitchkit.tuner.harmony.btc

import android.util.Log
import com.nicos.pitchkit.BuildConfig
import com.nicos.pitchkit.tuner.harmony.ChordRecognition
import com.nicos.pitchkit.tuner.harmony.ChordRecognizer
import com.nicos.pitchkit.tuner.harmony.ChordStabilizer
import com.nicos.pitchkit.tuner.harmony.PitchClassChordReranker
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetPostProcessor
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtChordTemplateDetector
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtCpuFrontend
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtPlanDecoder
import com.nicos.pitchkit.tuner.harmony.chordnet.FloatRingBuffer
import com.nicos.pitchkit.tuner.harmony.chordnet.LivePitchEvidenceAccumulator
import com.nicos.pitchkit.tuner.harmony.chordnet.StreamingPcmResampler
import com.nicos.pitchkit.tuner.models.AudioFrame
import java.security.MessageDigest

/**
 * Experimental rolling use of the BTC continual-learning checkpoint.
 *
 * BTC was trained with 108-frame bidirectional windows, so this path pads the
 * unavailable future context during startup. It is intentionally explicit and
 * never selected by Auto until real-device latency/accuracy justify it.
 */
class BtcStreamingRecognizer(
    modelBytes: ByteArray,
    metadata: BtcMetadata,
    cqtPlanBytes: ByteArray,
    referenceA4Hz: Double = 440.0,
    minimumConfidence: Double = 0.08,
) : ChordRecognizer {
    private companion object {
        const val LIVE_CONTEXT_FRAMES = 40
        const val STARTUP_FRAMES = 10
        const val INFERENCE_STRIDE_FRAMES = 2
        const val LIVE_SMOOTHING_KERNEL = 5
        const val DSP_OVERRIDE_MODEL_CONFIDENCE = 0.80
    }

    private val runner = BtcOnnxRunner(modelBytes, metadata.modelSha256)
    private val metadata = metadata
    private val plan = CqtPlanDecoder.decodeAndVerify(cqtPlanBytes)
    private val frontend: CqtCpuFrontend
    private val resampler = StreamingPcmResampler(
        targetRate = BtcContract.SAMPLE_RATE,
        pitchScale = 440.0 / referenceA4Hz,
    )
    private val audio = FloatRingBuffer(LIVE_CONTEXT_FRAMES * BtcContract.HOP_LENGTH)
    private val gestureEvidence = LivePitchEvidenceAccumulator()
    private val stabilizer = ChordStabilizer(
        minimumConfidence = minimumConfidence,
        changeConfirmations = 2,
    )

    private var totalTargetSamples = 0L
    private var lastInferenceAt = 0L
    private var closed = false

    init {
        require(referenceA4Hz in 300.0..600.0)
        require(sha256(cqtPlanBytes) == BtcContract.PLAN_SHA256) {
            "Unexpected BTC CQT plan SHA-256"
        }
        require(plan.config.sampleRate.toInt() == BtcContract.SAMPLE_RATE)
        require(plan.config.hopLength == BtcContract.HOP_LENGTH)
        require(plan.config.nBins == BtcContract.INPUT_BINS)
        require(plan.config.logMagnitude)
        frontend = CqtCpuFrontend(plan)
    }

    @Synchronized
    override fun recognize(frame: AudioFrame): ChordRecognition? {
        if (closed) return null
        val target = resampler.process(frame.toMono(), frame.sampleRate)
        if (target.isEmpty()) return stabilizer.currentWithoutPrediction()
        audio.append(target)
        totalTargetSamples += target.size

        val minimumSamples = (STARTUP_FRAMES - 1) * BtcContract.HOP_LENGTH
        if (audio.size < minimumSamples) return stabilizer.currentWithoutPrediction()
        val inferenceStride = INFERENCE_STRIDE_FRAMES * BtcContract.HOP_LENGTH
        if (totalTargetSamples - lastInferenceAt < inferenceStride) {
            return stabilizer.currentWithoutPrediction()
        }
        lastInferenceAt = totalTargetSamples

        val features = frontend.transform(audio.toFloatArray())
        if (features.frameCount <= 0 || closed) return stabilizer.update(null)
        val validFrames = features.frameCount.coerceAtMost(BtcContract.SEQUENCE_LENGTH)
        val modelInput = FloatArray(BtcContract.SEQUENCE_LENGTH * BtcContract.INPUT_BINS)
        val denominator = metadata.std.coerceAtLeast(1e-8f)

        val sourceFirst = (features.frameCount - validFrames).coerceAtLeast(0)
        for (frameIndex in 0 until validFrames) {
            val sourceOffset = (sourceFirst + frameIndex) * BtcContract.INPUT_BINS
            val destinationOffset = frameIndex * BtcContract.INPUT_BINS
            for (bin in 0 until BtcContract.INPUT_BINS) {
                modelInput[destinationOffset + bin] =
                    (features.values[sourceOffset + bin] - metadata.mean) / denominator
            }
        }

        val logits = runner.infer(modelInput, windowCount = 1)
        if (closed) return null
        val predictions = ChordNetPostProcessor.decode(
            logits = logits,
            windowCount = 1,
            validFrameCount = validFrames,
            smoothingKernel = LIVE_SMOOTHING_KERNEL,
            includeAlternatives = BuildConfig.DEBUG,
        )
        val prediction = predictions[validFrames - 1]

        val currentPitch = PitchClassChordReranker.cqtEvidence(
            values = features.values,
            frameCount = features.frameCount,
            binCount = features.binCount,
            fmin = plan.config.fmin,
            binsPerOctave = plan.config.binsPerOctave,
            logMagnitude = plan.config.logMagnitude,
            tailFrames = 4,
        )
        val currentBass = PitchClassChordReranker.cqtBassEvidence(
            values = features.values,
            frameCount = features.frameCount,
            binCount = features.binCount,
            fmin = plan.config.fmin,
            binsPerOctave = plan.config.binsPerOctave,
            logMagnitude = plan.config.logMagnitude,
            tailFrames = 4,
        )
        val gesture = gestureEvidence.update(currentPitch, currentBass)
        if (!gesture.ready) {
            if (BuildConfig.DEBUG) {
                Log.d(
                    "PitchKitChord",
                    "backend=BTC-Live forming updates=${gesture.updateCount} " +
                        "stable=${gesture.stableUpdates} raw=${prediction.displayLabel ?: "N"}",
                )
            }
            return stabilizer.currentWithoutPrediction()
        }

        val reranked = prediction.displayLabel?.let {
            PitchClassChordReranker.rerank(it, gesture.pitch)
        }
        val rootResolved = reranked?.let {
            PitchClassChordReranker.resolveEquivalentRoot(
                label = it.label,
                pitchEvidence = gesture.pitch,
                bassEvidence = gesture.bass,
            )
        }
        val modelLabel = rootResolved?.label ?: reranked?.label ?: prediction.displayLabel

        val dsp = CqtChordTemplateDetector.detect(gesture.pitch, gesture.bass)
        val strongDsp = dsp != null &&
            CqtChordTemplateDetector.isRescueCandidate(dsp.label) &&
            dsp.score >= 0.58 && dsp.margin >= 0.022
        val useDsp = when {
            dsp == null -> false
            modelLabel == null -> strongDsp
            dsp.label == modelLabel -> false
            !strongDsp -> false
            prediction.confidence >= DSP_OVERRIDE_MODEL_CONFIDENCE -> false
            else -> true
        }

        val finalLabel = if (useDsp) dsp!!.label else modelLabel
        val finalConfidence = if (useDsp) {
            dsp!!.score.coerceIn(0.0, 1.0)
        } else {
            prediction.confidence
        }
        val raw = finalLabel?.let {
            ChordRecognition(
                label = it,
                confidence = finalConfidence,
                backend = if (useDsp) {
                    "BTC Live Experimental + temporal CQT DSP"
                } else {
                    "BTC Live Experimental"
                },
            )
        }
        val emitted = stabilizer.update(raw)

        if (BuildConfig.DEBUG) {
            val top = prediction.alternatives.joinToString(" | ") { candidate ->
                "${candidate.displayLabel ?: candidate.rawLabel}=${"%.3f".format(candidate.confidence)}"
            }
            val correction = if (useDsp && dsp != null) {
                " correction=[dsp=${modelLabel ?: "N"}->${dsp.label} " +
                    "score=${"%.3f".format(dsp.score)} margin=${"%.3f".format(dsp.margin)}]"
            } else {
                ""
            }
            Log.d(
                "PitchKitChord",
                "backend=BTC-Live raw=${prediction.displayLabel ?: "N"} " +
                    "conf=${"%.3f".format(prediction.confidence)} " +
                    "context=$validFrames/${BtcContract.SEQUENCE_LENGTH} " +
                    "gesture=${gesture.updateCount}/${gesture.stableUpdates} " +
                    "top3=[$top]$correction emitted=${emitted?.label ?: "-"}",
            )
        }
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
        runner.close()
        resetState()
    }

    private fun resetState() {
        audio.clear()
        resampler.reset()
        totalTargetSamples = 0L
        lastInferenceAt = 0L
        gestureEvidence.reset()
        stabilizer.reset()
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
