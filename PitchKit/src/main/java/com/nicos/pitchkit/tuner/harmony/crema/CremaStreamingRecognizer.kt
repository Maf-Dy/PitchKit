package com.nicos.pitchkit.tuner.harmony.crema

import android.os.SystemClock
import android.util.Log
import com.nicos.pitchkit.BuildConfig
import com.nicos.pitchkit.tuner.harmony.ChordRecognition
import com.nicos.pitchkit.tuner.harmony.ChordRecognizer
import com.nicos.pitchkit.tuner.harmony.ChordStabilizer
import com.nicos.pitchkit.tuner.harmony.PitchClassChordReranker
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtChordTemplateDetector
import com.nicos.pitchkit.tuner.harmony.chordnet.FloatRingBuffer
import com.nicos.pitchkit.tuner.harmony.chordnet.LivePitchEvidenceAccumulator
import com.nicos.pitchkit.tuner.harmony.chordnet.StreamingPcmResampler
import com.nicos.pitchkit.tuner.models.AudioFrame

/** Low-latency rolling Crema 0.2.0 recognizer with temporal chord-gesture fusion. */
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
        const val EVIDENCE_SMOOTHING_FRAMES = 3
        const val DSP_OVERRIDE_MODEL_CONFIDENCE = 0.80

        val JAZZ_RESCUE_SUFFIXES = listOf(
            "6/9", "m6", "dim7", "ø7", "m9", "9",
        )
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
    private val gestureEvidence = LivePitchEvidenceAccumulator()

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
        if (closed) return null
        val onnxDoneAt = SystemClock.elapsedRealtimeNanos()
        val prediction = decoder.decode(heads)

        logTiming(startedAt, hcqtDoneAt, onnxDoneAt, features.frameCount)

        val currentPitchEvidence = averageTail(
            values = heads.pitch,
            width = CremaContract.PITCH_COUNT,
            frames = heads.frames,
            tailFrames = EVIDENCE_SMOOTHING_FRAMES,
            outputWidth = 12,
        )
        val currentBassEvidence = averageTail(
            values = heads.bass,
            width = CremaContract.BASS_COUNT,
            frames = heads.frames,
            tailFrames = EVIDENCE_SMOOTHING_FRAMES,
            outputWidth = 12,
        )
        val gesture = gestureEvidence.update(currentPitchEvidence, currentBassEvidence)

        if (!gesture.ready) {
            if (BuildConfig.DEBUG) {
                Log.d(
                    "PitchKitChord",
                    "backend=Crema forming updates=${gesture.updateCount} " +
                        "stable=${gesture.stableUpdates} raw=${prediction?.label ?: "N"}",
                )
            }
            return stabilizer.currentWithoutPrediction()
        }

        val reranked = prediction?.label?.let {
            PitchClassChordReranker.rerank(it, gesture.pitch)
        }
        val rootResolved = reranked?.let {
            PitchClassChordReranker.resolveEquivalentRoot(
                label = it.label,
                pitchEvidence = gesture.pitch,
                bassEvidence = gesture.bass,
            )
        }
        val modelLabel = rootResolved?.label ?: reranked?.label ?: prediction?.label

        val dsp = CqtChordTemplateDetector.detect(
            pitchEvidence = gesture.pitch,
            bassEvidence = gesture.bass,
        )
        val strongJazzDsp = dsp != null &&
            isJazzRescueLabel(dsp.label) &&
            dsp.score >= 0.58 &&
            dsp.margin >= 0.022
        val useDsp = when {
            dsp == null -> false
            modelLabel == null -> strongJazzDsp
            dsp.label == modelLabel -> false
            !strongJazzDsp -> false
            (prediction?.confidence ?: 0.0) >= DSP_OVERRIDE_MODEL_CONFIDENCE -> false
            else -> true
        }

        val finalLabel = if (useDsp) dsp!!.label else modelLabel
        val finalConfidence = if (useDsp) {
            maxOf(prediction?.confidence ?: 0.0, dsp!!.score.coerceIn(0.0, 1.0))
        } else {
            prediction?.confidence ?: 0.0
        }

        val rawRecognition = finalLabel?.let {
            ChordRecognition(
                label = it,
                confidence = finalConfidence,
                backend = if (useDsp) "Crema + temporal DSP" else "Crema 0.2.0",
            )
        }
        val emitted = stabilizer.update(rawRecognition)
        logDecision(
            prediction = prediction,
            reranked = reranked,
            rootResolved = rootResolved,
            dsp = dsp,
            useDsp = useDsp,
            gestureUpdates = gesture.updateCount,
            gestureStable = gesture.stableUpdates,
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
        gestureEvidence.reset()
        stabilizer.reset()
        inferenceCount = 0
    }

    private fun averageTail(
        values: FloatArray,
        width: Int,
        frames: Int,
        tailFrames: Int,
        outputWidth: Int,
    ): FloatArray {
        if (frames <= 0 || width <= 0 || values.size < frames * width) return FloatArray(outputWidth)
        val count = tailFrames.coerceIn(1, frames)
        val first = frames - count
        val result = FloatArray(outputWidth)
        for (frame in first until frames) {
            val offset = frame * width
            for (index in 0 until outputWidth.coerceAtMost(width)) {
                result[index] += values[offset + index]
            }
        }
        for (index in result.indices) result[index] /= count.toFloat()
        val peak = result.maxOrNull()?.coerceAtLeast(0f) ?: 0f
        if (peak > 1e-8f) {
            for (index in result.indices) result[index] = (result[index] / peak).coerceIn(0f, 1f)
        }
        return result
    }

    private fun isJazzRescueLabel(label: String): Boolean =
        JAZZ_RESCUE_SUFFIXES.any { suffix -> label.endsWith(suffix) }

    private fun logDecision(
        prediction: CremaDecodedChord?,
        reranked: com.nicos.pitchkit.tuner.harmony.PitchClassRerankResult?,
        rootResolved: com.nicos.pitchkit.tuner.harmony.PitchClassRerankResult?,
        dsp: CqtChordTemplateDetector.Result?,
        useDsp: Boolean,
        gestureUpdates: Int,
        gestureStable: Int,
        emitted: ChordRecognition?,
    ) {
        if (!BuildConfig.DEBUG) return
        val top = prediction?.alternatives
            ?.joinToString(separator = " | ") { candidate ->
                "${candidate.label}=fit:${"%.3f".format(candidate.fit)},tag:${"%.3f".format(candidate.tagConfidence)}"
            }
            .orEmpty()
        val corrections = mutableListOf<String>()
        reranked?.takeIf { it.changed }?.let { corrections += "quality=${prediction?.label}->${it.label}" }
        rootResolved?.takeIf { it.changed }?.let { corrections += "root=${reranked?.label}->${it.label}" }
        if (useDsp && dsp != null) {
            corrections += "dsp=${rootResolved?.label ?: reranked?.label ?: prediction?.label ?: "N"}->${dsp.label} " +
                "score=${"%.3f".format(dsp.score)} margin=${"%.3f".format(dsp.margin)}"
        }
        val correctionText = if (corrections.isEmpty()) "" else " correction=[${corrections.joinToString(",")}]"
        Log.d(
            "PitchKitChord",
            "backend=Crema raw=${prediction?.label ?: "N"} model=${prediction?.rawLabel ?: "N"} " +
                "conf=${prediction?.confidence?.let { "%.3f".format(it) } ?: "-"} " +
                "gesture=$gestureUpdates/$gestureStable top3=[$top]$correctionText " +
                "emitted=${emitted?.label ?: "-"}",
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
