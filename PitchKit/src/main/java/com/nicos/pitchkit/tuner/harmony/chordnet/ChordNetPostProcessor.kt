package com.nicos.pitchkit.tuner.harmony.chordnet

import kotlin.math.exp

object ChordNetPostProcessor {
    data class FramePrediction(
        val frameIndex: Int,
        val labelIndex: Int,
        val rawLabel: String,
        val displayLabel: String?,
        val confidence: Double,
    )

    fun decode(
        logits: FloatArray,
        windowCount: Int,
        validFrameCount: Int,
        smoothingKernel: Int = ChordNetContract.SMOOTHING_KERNEL,
    ): List<FramePrediction> {
        require(windowCount > 0)
        require(smoothingKernel > 0 && smoothingKernel % 2 == 1) {
            "smoothingKernel must be a positive odd number"
        }
        val totalFrames = windowCount * ChordNetContract.SEQUENCE_LENGTH
        require(validFrameCount in 1..totalFrames)
        require(logits.size == totalFrames * ChordNetContract.CHORD_COUNT)

        val radius = (smoothingKernel - 1) / 2
        return List(validFrameCount) { frame ->
            val first = (frame - radius).coerceAtLeast(0)
            val last = (frame + radius).coerceAtMost(validFrameCount - 1)
            val smoothed = DoubleArray(ChordNetContract.CHORD_COUNT)
            val count = last - first + 1

            for (neighbor in first..last) {
                val offset = neighbor * ChordNetContract.CHORD_COUNT
                for (chord in smoothed.indices) {
                    smoothed[chord] += logits[offset + chord].toDouble()
                }
            }
            for (chord in smoothed.indices) smoothed[chord] /= count

            var bestIndex = 0
            for (chord in 1 until smoothed.size) {
                if (smoothed[chord] > smoothed[bestIndex]) bestIndex = chord
            }

            val maxLogit = smoothed[bestIndex]
            var denominator = 0.0
            for (value in smoothed) denominator += exp(value - maxLogit)
            val confidence = if (denominator > 0.0) 1.0 / denominator else 0.0

            FramePrediction(
                frameIndex = frame,
                labelIndex = bestIndex,
                rawLabel = ChordNetVocabulary.labelAt(bestIndex),
                displayLabel = ChordNetVocabulary.displayLabel(bestIndex),
                confidence = confidence,
            )
        }
    }
}
