package com.nicos.pitchkit.tuner.harmony.chordnet

import kotlin.math.max

/**
 * Direct pitch-class fallback for live recognition.
 *
 * The neural recognizer remains primary. This detector is only allowed to
 * override it when the observed pitch set strongly supports a chord spelling
 * which the 170-class model cannot represent directly (9ths, altered dominant
 * colors, 11ths/13ths, etc.) or resolves a known ambiguous quality.
 */
internal object CqtChordTemplateDetector {
    data class Result(
        val label: String,
        val score: Double,
        val margin: Double,
    )

    private data class Quality(
        val suffix: String,
        val required: IntArray,
        val optional: IntArray = intArrayOf(),
        val minimumColorSupport: Float = 0.18f,
        val rescue: Boolean = false,
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
        Quality("m6", intArrayOf(0, 3, 7, 9), rescue = true),
        Quality("dim", intArrayOf(0, 3, 6)),
        Quality("dim7", intArrayOf(0, 3, 6, 9), rescue = true),
        Quality("ø7", intArrayOf(0, 3, 6, 10), rescue = true),
        Quality("sus2", intArrayOf(0, 2, 7)),
        Quality("sus4", intArrayOf(0, 5, 7)),

        // The fifth is optional in the extended voicings below. Requiring the
        // harmonic identity tones (3rd/7th/color) matches normal piano/guitar
        // practice better than demanding every textbook chord tone.
        Quality("6/9", intArrayOf(0, 4, 9, 2), optional = intArrayOf(7), minimumColorSupport = 0.20f, rescue = true),
        Quality("9", intArrayOf(0, 4, 10, 2), optional = intArrayOf(7), minimumColorSupport = 0.20f, rescue = true),
        Quality("maj9", intArrayOf(0, 4, 11, 2), optional = intArrayOf(7), minimumColorSupport = 0.20f, rescue = true),
        Quality("m9", intArrayOf(0, 3, 10, 2), optional = intArrayOf(7), minimumColorSupport = 0.20f, rescue = true),
        Quality("7♭9", intArrayOf(0, 4, 10, 1), optional = intArrayOf(7), minimumColorSupport = 0.22f, rescue = true),
        Quality("7♯9", intArrayOf(0, 4, 10, 3), optional = intArrayOf(7), minimumColorSupport = 0.22f, rescue = true),
        Quality("7♯11", intArrayOf(0, 4, 10, 6), optional = intArrayOf(2, 7), minimumColorSupport = 0.22f, rescue = true),
        Quality("7♭13", intArrayOf(0, 4, 10, 8), optional = intArrayOf(2, 7), minimumColorSupport = 0.22f, rescue = true),
        Quality("m11", intArrayOf(0, 3, 10, 5), optional = intArrayOf(2, 7), minimumColorSupport = 0.22f, rescue = true),
        Quality("13", intArrayOf(0, 4, 10, 9), optional = intArrayOf(2, 7), minimumColorSupport = 0.22f, rescue = true),
        Quality("maj13", intArrayOf(0, 4, 11, 9), optional = intArrayOf(2, 7), minimumColorSupport = 0.22f, rescue = true),
        Quality("m13", intArrayOf(0, 3, 10, 9), optional = intArrayOf(2, 7), minimumColorSupport = 0.22f, rescue = true),
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

    fun isRescueCandidate(label: String): Boolean = qualities.any { quality ->
        quality.rescue && label.endsWith(quality.suffix)
    }

    private fun supported(root: Int, quality: Quality, pitch: FloatArray): Boolean {
        val values = quality.required.map { interval -> pitch[(root + interval) % 12] }
        if (values.any { it < 0.14f }) return false

        // For complex qualities, the final required tone is the defining color
        // tone (9/b9/#9/#11/11/13/b13). It must be more than leakage.
        if (quality.rescue && quality.required.size >= 4) {
            val color = values.last()
            if (color < quality.minimumColorSupport) return false
        }
        return true
    }

    private fun score(
        root: Int,
        quality: Quality,
        pitch: FloatArray,
        bass: FloatArray,
    ): Double {
        val requiredPcs = quality.required.map { (root + it) % 12 }
        val optionalPcs = quality.optional.map { (root + it) % 12 }
        val required = requiredPcs.map { pitch[it].toDouble() }
        val mean = required.average()
        val weakest = required.minOrNull() ?: 0.0
        val expectedSet = (requiredPcs + optionalPcs).toSet()

        var strongestOutside = 0.0
        for (pc in 0 until 12) {
            if (pc !in expectedSet) strongestOutside = max(strongestOutside, pitch[pc].toDouble())
        }

        val optionalSupport = if (optionalPcs.isEmpty()) {
            0.0
        } else {
            optionalPcs.map { pitch[it].toDouble() }.average()
        }

        var strongestChordToneBass = 0.0
        for (pc in expectedSet) strongestChordToneBass = max(strongestChordToneBass, bass[pc].toDouble())
        val rootBass = bass[root].toDouble()

        val complexity = (quality.required.size + quality.optional.size - 3).coerceAtLeast(0)
        return 0.52 * mean +
            0.27 * weakest +
            0.05 * optionalSupport -
            0.17 * strongestOutside +
            0.18 * rootBass +
            0.03 * strongestChordToneBass -
            0.008 * complexity
    }

    private fun normalize(values: FloatArray): FloatArray {
        val peak = values.maxOrNull()?.coerceAtLeast(0f) ?: 0f
        if (peak <= 1e-8f) return FloatArray(12)
        return FloatArray(12) { index -> (values[index] / peak).coerceIn(0f, 1f) }
    }
}
