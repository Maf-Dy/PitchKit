package com.nicos.pitchkit.tuner.harmony.lvchordia

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LvSongDictionaryPitchRerankerTest {
    @Test
    fun gSixOverNineBeatsGSevenWhenEAndAArePresentAndFIsAbsent() {
        val gMajorTriad = 8 // N=0, then C:maj=1 ... G:maj=8
        val candidates = listOf(
            candidate("G7", gMajorTriad, seventh = 2),
            candidate("G", gMajorTriad),
            candidate("G6/9", gMajorTriad, ninth = 1, thirteenth = 1),
        )
        val harmony = evidence(7, 11, 2, 4, 9) // G B D E A
        val bass = evidence(7)

        val result = LvSongDictionaryPitchReranker.rerank(
            label = "G7",
            candidates = candidates,
            harmonyEvidence = harmony,
            bassEvidence = bass,
        )

        assertTrue(result.changed)
        assertEquals("G6/9", result.label)
    }

    @Test
    fun realGSevenStaysGSeven() {
        val gMajorTriad = 8
        val candidates = listOf(
            candidate("G7", gMajorTriad, seventh = 2),
            candidate("G", gMajorTriad),
            candidate("G6/9", gMajorTriad, ninth = 1, thirteenth = 1),
        )

        val result = LvSongDictionaryPitchReranker.rerank(
            label = "G7",
            candidates = candidates,
            harmonyEvidence = evidence(7, 11, 2, 5), // G B D F
            bassEvidence = evidence(7),
        )

        assertEquals("G7", result.label)
    }

    private fun candidate(
        label: String,
        triad: Int,
        seventh: Int = 0,
        ninth: Int = 0,
        eleventh: Int = 0,
        thirteenth: Int = 0,
    ) = LvChordiaDictionaryCandidate(
        rawLabel = label,
        displayLabel = label,
        triad = triad,
        bass = 7,
        seventh = seventh,
        ninth = ninth,
        eleventh = eleventh,
        thirteenth = thirteenth,
    )

    private fun evidence(vararg pcs: Int): FloatArray = FloatArray(12) { pc ->
        if (pc in pcs) 1f else 0f
    }
}
