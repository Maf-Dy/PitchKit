package com.nicos.pitchkit.tuner.harmony.lvchordia

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LvChordiaDictionaryDiagnosticsTest {
    @Test
    fun ranksHalfDiminishedFromDecomposedHeads() {
        val dictionary = LvChordiaDictionary(
            transitionPenalty = 30.0,
            candidates = listOf(
                LvChordiaDictionaryCandidate(
                    rawLabel = "N",
                    displayLabel = null,
                    triad = 0,
                    bass = -1,
                    seventh = -1,
                    ninth = -1,
                    eleventh = -1,
                    thirteenth = -1,
                ),
                LvChordiaDictionaryCandidate(
                    rawLabel = "F#:hdim7",
                    displayLabel = "F#ø7",
                    triad = 55,
                    bass = 6,
                    seventh = 2,
                    ninth = 0,
                    eleventh = 0,
                    thirteenth = 0,
                ),
            ),
        )
        val decoder = LvChordiaDictionaryDecoder(dictionary)

        val heads = LvChordiaHeads(
            frames = 1,
            triad = FloatArray(LvChordiaContract.TRIAD_COUNT) { 0.001f }.also { it[55] = 0.90f },
            bass = FloatArray(LvChordiaContract.BASS_COUNT) { 0.001f }.also { it[7] = 0.90f },
            seventh = floatArrayOf(0.02f, 0.02f, 0.90f, 0.06f),
            ninth = floatArrayOf(0.90f, 0.05f, 0.03f, 0.02f),
            eleventh = floatArrayOf(0.90f, 0.05f, 0.05f),
            thirteenth = floatArrayOf(0.90f, 0.05f, 0.05f),
        )

        val diagnostic = decoder.diagnoseFrames(heads, intArrayOf(0), limit = 2).single()

        assertEquals("F#:hdim7", diagnostic.topCandidates.first().rawLabel)
        assertEquals("F#ø7", diagnostic.topCandidates.first().displayLabel)
        assertTrue(diagnostic.headSummary.contains("F#:dim=0.900"))
        assertTrue(diagnostic.headSummary.contains("b7=0.900"))
    }
}
