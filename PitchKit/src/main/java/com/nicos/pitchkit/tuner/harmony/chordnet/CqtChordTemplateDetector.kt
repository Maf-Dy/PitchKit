package com.nicos.pitchkit.tuner.harmony.chordnet

import kotlin.math.max

/**
 * Direct pitch-class fallback for live recognition.
 *
 * ChordNet remains the primary recognizer. This detector is intentionally
 * conservative and is only used when its pitch-set evidence is strong enough
 * to rescue a low-confidence/no-chord neural result. Bass evidence breaks the
 * m6 <-> hdim7 equivalence and the four-way symmetry of dim7.
 */
internal object CqtChordTemplateDetector {
    data class Result(
        val label: String,
        val score: Double,
        val margin: Double,
    )

    private data class Quality(
        val suffix: String,
        val intervals: IntArray,
        val minimumExtensionSupport: Float = 0.18f,
    )

    private val sharpNames = arrayOf(
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    )

    private val qualities = listOf(
        Quality("", intArrayOf(0, 4, 7)),
        Quality("m", intArrayOf(0, 3, 7)),
        Quality("7", intArrayOf(0, 4, 7, 10)),
        Quality("maj7", intArrayOf(0, 4, 7, 11)),
        Quality("m7", intArrayOf(0, 3, 7, 10)),
        Quality("6", intArrayOf(0, 4, 7, 9)),
        Quality("m6", intArrayOf(0, 3, 7, 9)),
        Quality("dim", intArrayOf(0, 3, 6)),
        Quality("dim7", intArrayOf(0, 3, 6, 9)),
        Quality("ø7", intArrayOf(0, 3, 6, 10)),
        Quality("sus2", intArrayOf(0, 2, 7)),
        Quality("sus4", intArrayOf(0, 5, 7)),
        Quality("6/9", intArrayOf(0, 2, 4, 7, 9), minimumExtensionSupport = 0.20f),
        Quality("9", intArrayOf(0, 2, 4, 7, 10), minimumExtensionSupport = 0.20f),
        Quality("m9", intArrayOf(0, 2, 3, 7, 10), minimumExtensionSupport = 0.20f),
    )

    fun detect(pitchEvidence: FloatArray, bassEvidence: FloatArray): Result? {
        require(pitchEvidence.size == 12)
        require(bassEvidence.size == 12)

        val pitch = normalize(pitchEvidence)
        val bass = normalize(bassEvidence)
        if ((pitch.maxOrNull() ?: 0f) <= 0f) return null

        var bestLabel: String? = null
        var bestScore = Double.NEGATIVE_INFINITY
        var secondScore = Double.NEGATIVE_INFINITY

        for (root in 0 until 12) {
            for (quality in qualities) {
                if (!supported(root, quality, pitch)) continue
                val score = score(root, quality, pitch, bass)
                if (score > bestScore) {
                    secondScore = bestScore
                    bestScore = score
                    bestLabel = sharpNames[root] + quality.suffix
                } else if (score > secondScore) {
                    secondScore = score
                }
            }
        }

        val label = bestLabel ?: return null
        val margin = if (secondScore.isFinite()) bestScore - secondScore else bestScore
        if (bestScore < 0.53 || margin < 0.018) return null
        return Result(label = label, score = bestScore, margin = margin)
    }

    private fun supported(root: Int, quality: Quality, pitch: FloatArray): Boolean {
        val values = quality.intervals.map { interval -> pitch[(root + interval) % 12] }
        if (values.any { it < 0.14f }) return false

        // Added tones must be real, not weak spectral leakage. The first three
        // intervals form the basic triad for every quality in this table except
        // sus/dim, where this remains harmless because there are no extensions.
        if (quality.intervals.size > 3) {
            for (index in 3 until quality.intervals.size) {
                if (values[index] < quality.minimumExtensionSupport) return false
            }
        }
        return true
    }

    private fun score(
        root: Int,
        quality: Quality,
        pitch: FloatArray,
        bass: FloatArray,
    ): Double {
        val requiredPcs = quality.intervals.map { (root + it) % 12 }
        val required = requiredPcs.map { pitch[it].toDouble() }
        val mean = required.average()
        val weakest = required.minOrNull() ?: 0.0
        val requiredSet = requiredPcs.toSet()

        var strongestOutside = 0.0
        for (pc in 0 until 12) {
            if (pc !in requiredSet) strongestOutside = max(strongestOutside, pitch[pc].toDouble())
        }

        var strongestChordToneBass = 0.0
        for (pc in requiredPcs) strongestChordToneBass = max(strongestChordToneBass, bass[pc].toDouble())
        val rootBass = bass[root].toDouble()

        val extensions = (quality.intervals.size - 3).coerceAtLeast(0)
        return 0.52 * mean +
            0.27 * weakest -
            0.15 * strongestOutside +
            0.18 * rootBass +
            0.03 * strongestChordToneBass -
            0.010 * extensions
    }

    private fun normalize(values: FloatArray): FloatArray {
        val peak = values.maxOrNull()?.coerceAtLeast(0f) ?: 0f
        if (peak <= 1e-8f) return FloatArray(12)
        return FloatArray(12) { index -> (values[index] / peak).coerceIn(0f, 1f) }
    }
}
