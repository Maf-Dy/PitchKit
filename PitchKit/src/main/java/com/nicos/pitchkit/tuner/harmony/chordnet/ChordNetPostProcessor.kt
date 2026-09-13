package com.nicos.pitchkit.tuner.harmony.chordnet

import kotlin.math.exp

object ChordNetPostProcessor {
    data class CandidatePrediction(
        val labelIndex: Int,
        val rawLabel: String,
        val displayLabel: String?,
        val confidence: Double,
    )

    data class FramePrediction(
        val frameIndex: Int,
        val labelIndex: Int,
        val rawLabel: String,
        val displayLabel: String?,
        val confidence: Double,
        val alternatives: List<CandidatePrediction>,
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

            val topIndices = topIndices(smoothed, 3)
            val bestIndex = topIndices.first()
            val maxLogit = smoothed[bestIndex]
            var denominator = 0.0
            for (value in smoothed) denominator += exp(value - maxLogit)
            val safeDenominator = denominator.coerceAtLeast(1e-12)

            fun probability(index: Int): Double =
                exp(smoothed[index] - maxLogit) / safeDenominator

            FramePrediction(
                frameIndex = frame,
                labelIndex = bestIndex,
                rawLabel = ChordNetVocabulary.labelAt(bestIndex),
                displayLabel = ChordNetVocabulary.displayLabel(bestIndex),
                confidence = probability(bestIndex),
                alternatives = topIndices.map { index ->
                    CandidatePrediction(
                        labelIndex = index,
                        rawLabel = ChordNetVocabulary.labelAt(index),
                        displayLabel = ChordNetVocabulary.displayLabel(index),
                        confidence = probability(index),
                    )
                },
            )
        }
    }

    private fun topIndices(values: DoubleArray, count: Int): List<Int> {
        val limit = count.coerceIn(1, values.size)
        val top = IntArray(limit) { -1 }
        for (index in values.indices) {
            for (slot in 0 until limit) {
                val current = top[slot]
                if (current < 0 || values[index] > values[current]) {
                    for (shift in limit - 1 downTo slot + 1) {
                        top[shift] = top[shift - 1]
                    }
                    top[slot] = index
                    break
                }
            }
        }
        return top.filter { it >= 0 }
    }
}
