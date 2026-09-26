package com.nicos.pitchkit.tuner.harmony.chordnet

import com.nicos.pitchkit.tuner.PitchDiagnostics as Log
import com.nicos.pitchkit.tuner.PitchDiagnostics
import com.nicos.pitchkit.tuner.harmony.ChordRecognition
import com.nicos.pitchkit.tuner.harmony.ChordUpdateTracker
import com.nicos.pitchkit.tuner.harmony.ChordRecognizer
import com.nicos.pitchkit.tuner.harmony.ChordStabilizer
import com.nicos.pitchkit.tuner.harmony.PitchClassChordReranker
import com.nicos.pitchkit.tuner.models.AudioFrame
import java.security.MessageDigest

/**
 * Low-latency rolling recognizer for ChordNet 2E1D.
 *
 * The temporal CQT/DSP rescue fusion is still here but is **off by default**
 * since the rescue ablation; see [dspRescue].
 */
class ChordNetStreamingRecognizer(
    modelBytes: ByteArray,
    planBytes: ByteArray,
    referenceA4Hz: Double = 440.0,
    minimumConfidence: Double = 0.08,
    /**
     * Whether the `CqtChordTemplateDetector` rescue lane may overrule the model.
     *
     * **Off by default since the rescue ablation**
     * (`.accuracy-work/annotations/live-rescue-ablation-report.md` §2, §6). On
     * GuitarSet comp-60 the lane is a net loss on real guitar: removing it gains
     * **+3.2** points of family accuracy (63.5 % → 66.7 %), drops the wrong-chord
     * rate 8.3 points (35.8 % → 27.5 %), roughly halves extension flicker
     * (91 → 48), raises performed-layer extension precision from **15.5 % to
     * 60.4 %** and lifts tension-*exact* 3.3 % → 5.3 %. The `A7♭9`-over-a-plain-
     * A-major flood — 2 195 frames — disappears entirely, and 4 of 5 styles win.
     * The one cost is blankness: 0.6 % → 5.8 % of frames, which is the better
     * failure and still well under Classic DSP's 12.5 %.
     *
     * `true` restores the pre-ablation behaviour. It is kept so the live replay
     * harness can still benchmark that configuration (`chordnet-rescue`,
     * `chordnet-rescue60`); nothing on a device selects it. `false` skips the
     * detector's own work rather than computing it and throwing it away.
     */
    private val dspRescue: Boolean = false,
    /**
     * Model confidence at or above which the rescue is never taken, when
     * [dspRescue] is on at all. Default is [DSP_OVERRIDE_MODEL_CONFIDENCE], now
     * 0.60: the ablation found the previously shipped 0.80 **dominated on every
     * measured axis** — family accuracy 63.5 % → 65.9 %, extension flicker
     * 91 → 68, TTSL p90 1.624 s → 1.537 s, performed extension precision
     * 15.5 % → 19.1 %, at identical 0.6 % blankness (report §6). Lowering it
     * further gates the rescue harder; raising it lets it fire on more confident
     * predictions.
     */
    private val rescueConfidenceThreshold: Double = DSP_OVERRIDE_MODEL_CONFIDENCE,
    /** Numeric stage timings only; does not change inference cadence or predictions. */
    private val stageTiming: ((String, Long) -> Unit)? = null,
) : ChordRecognizer {
    private val plan: CqtPlan
    private val frontend: CqtCpuFrontend
    private val runner: ChordNetOnnxRunner
    private val resampler: StreamingPcmResampler
    private val stabilizer = ChordStabilizer(
        minimumConfidence = minimumConfidence,
        changeConfirmations = 2,
    )
    private val gestureEvidence = LivePitchEvidenceAccumulator()

    internal companion object {
        const val LIVE_CONTEXT_FRAMES = 32
        const val STARTUP_FRAMES = 10
        const val INFERENCE_STRIDE_FRAMES = 2
        const val LIVE_SMOOTHING_KERNEL = 5
        /**
         * Default [rescueConfidenceThreshold] for when the rescue is enabled at
         * all. 0.60, not the historical 0.80, which the ablation report found
         * dominated on every measured axis (§6).
         */
        const val DSP_OVERRIDE_MODEL_CONFIDENCE = 0.60
    }

    private val maxSamples = LIVE_CONTEXT_FRAMES * ChordNetContract.HOP_LENGTH - 1
    private val audio = FloatRingBuffer(maxSamples)

    private var totalTargetSamples = 0L
    private var lastInferenceAt = 0L
    private var closed = false
    private val updates = ChordUpdateTracker()
    override val latestUpdate get() = updates.latest

    init {
        require(referenceA4Hz in 300.0..600.0) { "referenceA4Hz must be between 300 and 600 Hz" }
        require(rescueConfidenceThreshold in 0.0..1.0) {
            "rescueConfidenceThreshold must be between 0 and 1"
        }
        require(sha256(modelBytes) == ChordNetContract.MODEL_SHA256) {
            "Unexpected ChordNet model SHA-256"
        }
        require(sha256(planBytes) == ChordNetContract.PLAN_SHA256) {
            "Unexpected ChordNet CQT plan SHA-256"
        }
        plan = CqtPlanDecoder.decodeAndVerify(planBytes)
        frontend = CqtCpuFrontend(plan, stageTiming?.let { callback ->
            { stage, nanos -> callback("chordNet$stage", nanos) }
        })
        runner = ChordNetOnnxRunner(modelBytes)
        resampler = StreamingPcmResampler(
            pitchScale = 440.0 / referenceA4Hz,
        )
    }

    @Synchronized
    override fun recognize(frame: AudioFrame): ChordRecognition? {
        if (closed) return null

        val targetSamples = timed("chordNetResampleMs") { resampler.process(frame.toMono(), frame.sampleRate) }
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

        val features = timed("chordNetFrontendMs") { frontend.transform(audio.toFloatArray()) }
        if (features.frameCount <= 0 || closed) return updates.record(stabilizer.update(null))

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

        val logits = timed("chordNetOnnxMs") { runner.infer(modelInput, windowCount = 1) }
        if (closed) return null
        val predictions = timed("chordNetDecodeMs") { ChordNetPostProcessor.decode(
            logits = logits,
            windowCount = 1,
            validFrameCount = validFrames,
            smoothingKernel = LIVE_SMOOTHING_KERNEL,
            includeAlternatives = PitchDiagnostics.enabled,
        ) }

        val prediction = predictions[validFrames - 1]
        val currentPitchEvidence = PitchClassChordReranker.cqtEvidence(
            values = features.values,
            frameCount = features.frameCount,
            binCount = features.binCount,
            fmin = plan.config.fmin,
            binsPerOctave = plan.config.binsPerOctave,
            logMagnitude = plan.config.logMagnitude,
            tailFrames = 4,
        )
        val currentBassEvidence = PitchClassChordReranker.cqtBassEvidence(
            values = features.values,
            frameCount = features.frameCount,
            binCount = features.binCount,
            fmin = plan.config.fmin,
            binsPerOctave = plan.config.binsPerOctave,
            logMagnitude = plan.config.logMagnitude,
            tailFrames = 4,
        )
        val gesture = gestureEvidence.update(currentPitchEvidence, currentBassEvidence)
        val pitchEvidence = gesture.pitch
        val bassEvidence = gesture.bass

        val reranked = prediction.displayLabel?.let {
            PitchClassChordReranker.rerank(it, pitchEvidence)
        }
        val rootResolved = reranked?.let {
            PitchClassChordReranker.resolveEquivalentRoot(
                label = it.label,
                pitchEvidence = pitchEvidence,
                bassEvidence = bassEvidence,
            )
        }
        val modelLabel = rootResolved?.label ?: reranked?.label ?: prediction.displayLabel

        val dsp = if (dspRescue) {
            CqtChordTemplateDetector.detect(
                pitchEvidence = pitchEvidence,
                bassEvidence = bassEvidence,
            )
        } else {
            null
        }
        val strongDsp = dsp != null &&
            CqtChordTemplateDetector.isRescueCandidate(dsp.label) &&
            dsp.score >= 0.58 &&
            dsp.margin >= 0.022
        val useDsp = gesture.ready && when {
            dsp == null -> false
            modelLabel == null -> strongDsp
            dsp.label == modelLabel -> false
            !strongDsp -> false
            prediction.confidence >= rescueConfidenceThreshold -> false
            else -> true
        }

        // Do not publish a partial chord while new pitch classes are still being
        // added to the current gesture. This is the key arpeggio/together-chord
        // behavior: the gesture may resolve early when stable, but can collect
        // evidence for roughly a second while notes continue arriving.
        if (!gesture.ready) {
            updates.record(null)
            if (PitchDiagnostics.enabled) {
                Log.d(
                    "PitchKitChord",
                    "backend=ChordNet forming updates=${gesture.updateCount} " +
                        "stable=${gesture.stableUpdates} raw=${prediction.displayLabel ?: "N"}",
                )
            }
            return stabilizer.currentWithoutPrediction()
        }

        val finalLabel = if (useDsp) dsp!!.label else modelLabel
        val finalConfidence = if (useDsp) {
            // The neural confidence belongs to a different label. Do not attach
            // it to a DSP-rewritten chord; report the evidence for the label we
            // actually emit.
            dsp!!.score.coerceIn(0.0, 1.0)
        } else {
            prediction.confidence
        }

        val rawRecognition = finalLabel?.let {
            ChordRecognition(
                label = it,
                confidence = finalConfidence,
                backend = if (useDsp) "ChordNet + temporal CQT DSP" else "ChordNet 2E1D",
            )
        }
        val emitted = stabilizer.update(rawRecognition)

        if (PitchDiagnostics.enabled) {
            val top = prediction.alternatives.joinToString(separator = " | ") { candidate ->
                "${candidate.displayLabel ?: candidate.rawLabel}=${"%.3f".format(candidate.confidence)}"
            }
            val correctionParts = mutableListOf<String>()
            reranked?.takeIf { it.changed }?.let {
                correctionParts += "quality=${prediction.displayLabel}->${it.label}"
            }
            rootResolved?.takeIf { it.changed }?.let {
                correctionParts += "root=${reranked?.label}->${it.label} bass=${"%.3f".format(it.score)}"
            }
            if (useDsp && dsp != null) {
                correctionParts += "dsp=${modelLabel ?: "N"}->${dsp.label} score=${"%.3f".format(dsp.score)} margin=${"%.3f".format(dsp.margin)}"
            }
            val correction = if (correctionParts.isEmpty()) {
                ""
            } else {
                " correction=[${correctionParts.joinToString(",")}]"
            }
            Log.d(
                "PitchKitChord",
                "backend=ChordNet raw=${prediction.displayLabel ?: "N"} " +
                    "model=${prediction.rawLabel} conf=${"%.3f".format(prediction.confidence)} " +
                    "gesture=${gesture.updateCount}/${gesture.stableUpdates} " +
                    "top3=[$top]$correction emitted=${emitted?.label ?: "-"}",
            )
        }
        updates.record(emitted)
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
        try { runner.close() } finally { resetState() }
    }

    private fun resetState() {
        updates.reset()
        audio.clear()
        resampler.reset()
        totalTargetSamples = 0L
        lastInferenceAt = 0L
        gestureEvidence.reset()
        stabilizer.reset()
    }

    private inline fun <T> timed(stage: String, block: () -> T): T {
        val callback = stageTiming ?: return block()
        val started = System.nanoTime()
        return try { block() } finally { callback(stage, System.nanoTime() - started) }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
