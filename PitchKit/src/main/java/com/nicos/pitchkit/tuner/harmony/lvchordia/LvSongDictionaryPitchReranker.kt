package com.nicos.pitchkit.tuner.harmony.lvchordia

import kotlin.math.max

internal data class LvSongPitchRerankResult(
    val label: String,
    val changed: Boolean,
    val originalScore: Double,
    val score: Double,
)

/**
 * Direct-evidence refinement over LV Song's own full chord dictionary.
 *
 * LV Song remains responsible for the harmonic root/triad/context. Within that
 * family, the full dictionary competes on sustained mid/upper pitch evidence;
 * bass is intentionally only a weak cue so a loud bass line cannot dictate the
 * exact piano-harmony spelling.
 */
internal object LvSongDictionaryPitchReranker {
    private const val SWITCH_MARGIN = 0.050
    private const val MIN_EXTENSION_SUPPORT = 0.18f

    fun rerank(
        label: String,
        candidates: List<LvChordiaDictionaryCandidate>,
        harmonyEvidence: FloatArray,
        bassEvidence: FloatArray,
    ): LvSongPitchRerankResult {
        require(harmonyEvidence.size == 12)
        require(bassEvidence.size == 12)

        val base = candidates.firstOrNull { it.displayLabel == label }
            ?: return unchanged(label)
        val baseIdentity = triadIdentity(base.triad) ?: return unchanged(label)

        val family = candidates.filter { candidate ->
            candidate.displayLabel != null && triadIdentity(candidate.triad) == baseIdentity
        }
        if (family.isEmpty()) return unchanged(label)

        val baseScore = score(base, harmonyEvidence, bassEvidence)
        var best = base
        var bestScore = baseScore
        for (candidate in family) {
            if (!extensionsSupported(candidate, harmonyEvidence)) continue
            val candidateScore = score(candidate, harmonyEvidence, bassEvidence)
            if (candidateScore > bestScore) {
                best = candidate
                bestScore = candidateScore
            }
        }

        val bestLabel = best.displayLabel ?: label
        val changed = bestLabel != label && bestScore >= baseScore + SWITCH_MARGIN
        return LvSongPitchRerankResult(
            label = if (changed) bestLabel else label,
            changed = changed,
            originalScore = baseScore,
            score = if (changed) bestScore else baseScore,
        )
    }

    private fun score(
        candidate: LvChordiaDictionaryCandidate,
        harmony: FloatArray,
        bass: FloatArray,
    ): Double {
        val (root, triadType) = triadIdentity(candidate.triad) ?: return Double.NEGATIVE_INFINITY
        val coreIntervals = triadIntervals(triadType) ?: return Double.NEGATIVE_INFINITY
        val corePcs = coreIntervals.map { (root + it) % 12 }
        val extensionPcs = extensionPitchClasses(candidate, root)
        val allPcs = (corePcs + extensionPcs).toSet()

        var weightedCore = 0.0
        var coreWeight = 0.0
        for ((index, pc) in corePcs.withIndex()) {
            // Root/fifth may legitimately be omitted in dense jazz voicings;
            // the characteristic middle triad tone gets the strongest weight.
            val weight = when (index) {
                0 -> 0.55
                1 -> 1.00
                else -> 0.70
            }
            weightedCore += harmony[pc] * weight
            coreWeight += weight
        }
        val coreMean = if (coreWeight > 0.0) weightedCore / coreWeight else 0.0
        val extensionMean = if (extensionPcs.isEmpty()) {
            0.0
        } else {
            extensionPcs.map { harmony[it].toDouble() }.average()
        }

        var strongestOutside = 0.0
        for (pc in 0 until 12) {
            if (pc !in allPcs) strongestOutside = max(strongestOutside, harmony[pc].toDouble())
        }

        val bassMatch = if (candidate.bass in 0..11) bass[candidate.bass].toDouble() else 0.0
        val complexityPenalty = 0.010 * extensionPcs.size

        return 0.72 * coreMean +
            0.18 * extensionMean +
            0.08 * bassMatch -
            0.15 * strongestOutside -
            complexityPenalty
    }

    private fun extensionsSupported(
        candidate: LvChordiaDictionaryCandidate,
        harmony: FloatArray,
    ): Boolean {
        val (root, triadType) = triadIdentity(candidate.triad) ?: return false
        val core = triadIntervals(triadType)
            ?.map { harmony[(root + it) % 12] }
            ?: return false
        val extensions = extensionPitchClasses(candidate, root)
        if (extensions.isEmpty()) return true

        val coreMean = core.average().toFloat()
        val threshold = max(MIN_EXTENSION_SUPPORT, coreMean * 0.28f)
        return extensions.all { harmony[it] >= threshold }
    }

    private fun triadIdentity(encoded: Int): Pair<Int, Int>? {
        if (encoded <= 0) return null
        val zeroBased = encoded - 1
        val root = zeroBased % 12
        val triadType = zeroBased / 12 + 1
        if (triadType !in 1..6) return null
        return root to triadType
    }

    private fun triadIntervals(type: Int): IntArray? = when (type) {
        1 -> intArrayOf(0, 4, 7) // major
        2 -> intArrayOf(0, 3, 7) // minor
        3 -> intArrayOf(0, 5, 7) // sus4
        4 -> intArrayOf(0, 2, 7) // sus2
        5 -> intArrayOf(0, 3, 6) // diminished
        6 -> intArrayOf(0, 4, 8) // augmented
        else -> null
    }

    private fun extensionPitchClasses(
        candidate: LvChordiaDictionaryCandidate,
        root: Int,
    ): List<Int> {
        val result = mutableListOf<Int>()
        when (candidate.seventh) {
            1 -> result += (root + 11) % 12 // 7
            2 -> result += (root + 10) % 12 // b7
            3 -> result += (root + 9) % 12 // bb7
        }
        when (candidate.ninth) {
            1 -> result += (root + 2) % 12 // 9
            2 -> result += (root + 3) % 12 // #9
            3 -> result += (root + 1) % 12 // b9
        }
        when (candidate.eleventh) {
            1 -> result += (root + 5) % 12 // 11
            2 -> result += (root + 6) % 12 // #11
        }
        when (candidate.thirteenth) {
            1 -> result += (root + 9) % 12 // 13 / 6
            2 -> result += (root + 8) % 12 // b13
            3 -> result += (root + 7) % 12 // bb13, if present in a future dictionary
        }
        return result.distinct()
    }

    private fun unchanged(label: String) = LvSongPitchRerankResult(
        label = label,
        changed = false,
        originalScore = 0.0,
        score = 0.0,
    )
}
