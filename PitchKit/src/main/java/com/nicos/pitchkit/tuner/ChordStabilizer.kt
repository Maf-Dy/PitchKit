package com.nicos.pitchkit.tuner

import java.util.ArrayDeque

/** Stabilizes chord identity without pinning a new chord behind the old chord's score. */
internal class ChordStabilizer(
    private val windowSize: Int = 3,
    private val requiredAgreement: Int = 2,
    private val clearConfidence: Double = 0.04,
) {
    data class StableChord(val name: String, val score: Double, val confidence: Double)

    private val history = ArrayDeque<ChordDetector.ChordResult?>()
    private var stable: StableChord? = null

    fun accept(candidate: ChordDetector.ChordResult?): StableChord? {
        history.add(candidate)
        while (history.size > windowSize) history.removeFirst()

        if (history.count { it == null } >= requiredAgreement) stable = null

        val candidates = history.filterNotNull()
        val winner = candidates
            .groupBy { it.name }
            .maxByOrNull { (_, values) -> values.size }

        if (winner != null) {
            val latest = candidates.last { it.name == winner.key }
            val agreementNeeded = if (latest.confidence >= clearConfidence) {
                requiredAgreement
            } else {
                (requiredAgreement + 1).coerceAtMost(windowSize)
            }
            if (winner.value.size >= agreementNeeded) {
                stable = StableChord(latest.name, latest.score, latest.confidence)
            }
        }
        return stable
    }

    fun reset() {
        history.clear()
        stable = null
    }
}
