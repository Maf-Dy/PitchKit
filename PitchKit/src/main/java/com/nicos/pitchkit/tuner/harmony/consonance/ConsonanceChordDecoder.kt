package com.nicos.pitchkit.tuner.harmony.consonance

import kotlin.math.exp

internal data class ConsonanceDecodedFrame(
    val label: String?,
    val root: String?,
    val bass: String?,
    val pitchClasses: List<String>,
    val pitchProbabilities: FloatArray,
    val confidence: Double,
)

/** Decode root + bass + absolute pitch activations from Consonance Decomposed. */
internal class ConsonanceChordDecoder(
    private val preferFlats: Boolean = false,
    private val pitchThreshold: Float = ConsonanceContract.PITCH_THRESHOLD,
) {
    private data class Quality(
        val suffix: String,
        val intervals: Set<Int>,
    )

    private val sharpNames = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    private val flatNames = arrayOf("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

    private val qualities = listOf(
        Quality("6/9", setOf(0, 2, 4, 7, 9)),
        Quality("maj13", setOf(0, 2, 4, 7, 9, 11)),
        Quality("m13", setOf(0, 2, 3, 7, 9, 10)),
        Quality("13", setOf(0, 2, 4, 7, 9, 10)),
        Quality("maj9", setOf(0, 2, 4, 7, 11)),
        Quality("m9", setOf(0, 2, 3, 7, 10)),
        Quality("9", setOf(0, 2, 4, 7, 10)),
        Quality("m11", setOf(0, 2, 3, 5, 7, 10)),
        Quality("11", setOf(0, 2, 4, 5, 7, 10)),
        Quality("7b9", setOf(0, 1, 4, 7, 10)),
        Quality("7#9", setOf(0, 3, 4, 7, 10)),
        Quality("7#11", setOf(0, 4, 6, 7, 10)),
        Quality("7b13", setOf(0, 4, 7, 8, 10)),
        Quality("9#11", setOf(0, 2, 4, 6, 7, 10)),
        Quality("maj7", setOf(0, 4, 7, 11)),
        Quality("m(maj7)", setOf(0, 3, 7, 11)),
        Quality("m7", setOf(0, 3, 7, 10)),
        Quality("7", setOf(0, 4, 7, 10)),
        Quality("dim7", setOf(0, 3, 6, 9)),
        Quality("ø7", setOf(0, 3, 6, 10)),
        Quality("6", setOf(0, 4, 7, 9)),
        Quality("m6", setOf(0, 3, 7, 9)),
        Quality("sus2", setOf(0, 2, 7)),
        Quality("sus4", setOf(0, 5, 7)),
        Quality("dim", setOf(0, 3, 6)),
        Quality("aug", setOf(0, 4, 8)),
        Quality("m", setOf(0, 3, 7)),
        Quality("", setOf(0, 4, 7)),
    ).sortedByDescending { it.intervals.size }

    fun decode(heads: ConsonanceHeads): List<ConsonanceDecodedFrame> {
        if (heads.frames <= 0) return emptyList()
        require(heads.root.size == heads.frames * ConsonanceContract.ROOT_COUNT)
        require(heads.bass.size == heads.frames * ConsonanceContract.BASS_COUNT)
        require(heads.pitch.size == heads.frames * ConsonanceContract.PITCH_COUNT)

        val rootLogits = smoothCategorical(heads.root, heads.frames, ConsonanceContract.ROOT_COUNT)
        val bassLogits = smoothCategorical(heads.bass, heads.frames, ConsonanceContract.BASS_COUNT)
        val pitchLogits = smoothContinuous(heads.pitch, heads.frames, ConsonanceContract.PITCH_COUNT)

        return List(heads.frames) { frame ->
            decodeFrame(rootLogits, bassLogits, pitchLogits, frame)
        }
    }

    private fun decodeFrame(
        rootLogits: FloatArray,
        bassLogits: FloatArray,
        pitchLogits: FloatArray,
        frame: Int,
    ): ConsonanceDecodedFrame {
        val rootOffset = frame * ConsonanceContract.ROOT_COUNT
        val bassOffset = frame * ConsonanceContract.BASS_COUNT
        val pitchOffset = frame * ConsonanceContract.PITCH_COUNT

        val rootIndex = argmax(rootLogits, rootOffset, ConsonanceContract.ROOT_COUNT)
        val bassIndex = argmax(bassLogits, bassOffset, ConsonanceContract.BASS_COUNT)
        val rootProb = softmaxProbability(rootLogits, rootOffset, ConsonanceContract.ROOT_COUNT, rootIndex)
        val bassProb = softmaxProbability(bassLogits, bassOffset, ConsonanceContract.BASS_COUNT, bassIndex)

        val probabilities = FloatArray(12) { index -> sigmoid(pitchLogits[pitchOffset + index]) }
        if (rootIndex == 12) {
            return ConsonanceDecodedFrame(
                label = null,
                root = null,
                bass = null,
                pitchClasses = emptyList(),
                pitchProbabilities = probabilities,
                confidence = rootProb,
            )
        }

        val activeAbsolute = probabilities.indices
            .filter { probabilities[it] >= pitchThreshold }
            .toMutableSet()
        // Match the reference decoder's pragmatic completion: when root+third
        // are present but no fifth of any kind is active, add a perfect fifth.
        val rootRelative = activeAbsolute.map { floorMod12(it - rootIndex) }.toMutableSet()
        val hasThird = 4 in rootRelative || 3 in rootRelative
        val hasAnyFifth = 6 in rootRelative || 7 in rootRelative || 8 in rootRelative
        if (0 in rootRelative && hasThird && !hasAnyFifth) {
            activeAbsolute += floorMod12(rootIndex + 7)
            rootRelative += 7
        }

        val rootName = noteName(rootIndex)
        val bassName = bassIndex.takeIf { it in 0..11 }?.let(::noteName)
        val quality = qualities.firstOrNull { it.intervals == rootRelative }
        val coreLabel = if (quality != null) {
            rootName + quality.suffix
        } else {
            val degreeText = rootRelative
                .sorted()
                .joinToString(",") { intervalName(it) }
            if (degreeText.isBlank()) rootName else "$rootName:($degreeText)"
        }
        val label = if (bassIndex in 0..11 && bassIndex != rootIndex) {
            "$coreLabel/${noteName(bassIndex)}"
        } else {
            coreLabel
        }

        val pitchNames = activeAbsolute.sorted().map(::noteName)
        val activeConfidence = if (activeAbsolute.isEmpty()) {
            0.0
        } else {
            activeAbsolute.map { probabilities[it].toDouble() }.average()
        }
        val confidence = (0.45 * rootProb + 0.15 * bassProb + 0.40 * activeConfidence)
            .coerceIn(0.0, 1.0)

        return ConsonanceDecodedFrame(
            label = label,
            root = rootName,
            bass = bassName,
            pitchClasses = pitchNames,
            pitchProbabilities = probabilities,
            confidence = confidence,
        )
    }

    private fun smoothCategorical(values: FloatArray, frames: Int, width: Int): FloatArray =
        smoothContinuous(values, frames, width, window = 5)

    private fun smoothContinuous(
        values: FloatArray,
        frames: Int,
        width: Int,
        window: Int = 5,
    ): FloatArray {
        val output = FloatArray(values.size)
        val radius = window / 2
        for (frame in 0 until frames) {
            val first = (frame - radius).coerceAtLeast(0)
            val last = (frame + radius).coerceAtMost(frames - 1)
            val count = (last - first + 1).toFloat()
            val outOffset = frame * width
            for (sourceFrame in first..last) {
                val sourceOffset = sourceFrame * width
                for (index in 0 until width) {
                    output[outOffset + index] += values[sourceOffset + index] / count
                }
            }
        }
        return output
    }

    private fun argmax(values: FloatArray, offset: Int, width: Int): Int {
        var best = 0
        for (index in 1 until width) {
            if (values[offset + index] > values[offset + best]) best = index
        }
        return best
    }

    private fun softmaxProbability(
        values: FloatArray,
        offset: Int,
        width: Int,
        selected: Int,
    ): Double {
        var maxValue = Double.NEGATIVE_INFINITY
        for (index in 0 until width) maxValue = maxOf(maxValue, values[offset + index].toDouble())
        var denominator = 0.0
        for (index in 0 until width) denominator += exp(values[offset + index] - maxValue)
        return if (denominator > 0.0) exp(values[offset + selected] - maxValue) / denominator else 0.0
    }

    private fun sigmoid(value: Float): Float {
        val x = value.toDouble().coerceIn(-40.0, 40.0)
        return (1.0 / (1.0 + exp(-x))).toFloat()
    }

    private fun noteName(pc: Int): String =
        if (preferFlats) flatNames[floorMod12(pc)] else sharpNames[floorMod12(pc)]

    private fun intervalName(interval: Int): String = when (floorMod12(interval)) {
        0 -> "1"
        1 -> "b9"
        2 -> "9"
        3 -> "b3"
        4 -> "3"
        5 -> "11"
        6 -> "b5/#11"
        7 -> "5"
        8 -> "b13"
        9 -> "6/13"
        10 -> "b7"
        11 -> "7"
        else -> "?"
    }

    private fun floorMod12(value: Int): Int = ((value % 12) + 12) % 12
}
