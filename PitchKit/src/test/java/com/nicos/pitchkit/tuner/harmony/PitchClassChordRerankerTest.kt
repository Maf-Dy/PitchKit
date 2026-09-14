package com.nicos.pitchkit.tuner.harmony

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PitchClassChordRerankerTest {
    @Test
    fun reranksDominantSevenToSixNineWhenSixAndNineArePresent() {
        val evidence = evidence(
            "G" to 1.00f,
            "B" to 0.92f,
            "D" to 0.84f,
            "E" to 0.88f,
            "A" to 0.81f,
            "F" to 0.05f,
        )

        val result = PitchClassChordReranker.rerank("G7", evidence)

        assertTrue(result.changed)
        assertEquals("G6/9", result.label)
    }

    @Test
    fun keepsRealDominantSeven() {
        val evidence = evidence(
            "G" to 1.00f,
            "B" to 0.92f,
            "D" to 0.86f,
            "F" to 0.83f,
            "E" to 0.08f,
            "A" to 0.06f,
        )

        val result = PitchClassChordReranker.rerank("G7", evidence)

        assertFalse(result.changed)
        assertEquals("G7", result.label)
    }

    @Test
    fun upgradesDiminishedTriadToHalfDiminished() {
        val evidence = evidence(
            "F#" to 1.00f,
            "A" to 0.93f,
            "C" to 0.89f,
            "E" to 0.86f,
            "D#" to 0.05f,
        )

        val result = PitchClassChordReranker.rerank("F#dim", evidence)

        assertTrue(result.changed)
        assertEquals("F#ø7", result.label)
    }

    @Test
    fun upgradesDiminishedTriadToFullyDiminishedSeven() {
        val evidence = evidence(
            "F#" to 1.00f,
            "A" to 0.93f,
            "C" to 0.89f,
            "D#" to 0.86f,
            "E" to 0.04f,
        )

        val result = PitchClassChordReranker.rerank("F#dim", evidence)

        assertTrue(result.changed)
        assertEquals("F#dim7", result.label)
    }

    @Test
    fun canMoveMinorFamilyToHalfDiminishedWhenFifthIsClearlyFlat() {
        val evidence = evidence(
            "F#" to 1.00f,
            "A" to 0.94f,
            "C" to 0.91f,
            "E" to 0.88f,
            "C#" to 0.04f,
        )

        val result = PitchClassChordReranker.rerank("F#m7", evidence)

        assertTrue(result.changed)
        assertEquals("F#ø7", result.label)
    }

    @Test
    fun resolvesAm6AsFSharpHalfDiminishedWhenFSharpOwnsTheBass() {
        val pitch = evidence(
            "F#" to 1.00f,
            "A" to 0.95f,
            "C" to 0.90f,
            "E" to 0.87f,
        )
        val bass = evidence(
            "F#" to 1.00f,
            "A" to 0.48f,
            "C" to 0.20f,
            "E" to 0.12f,
        )

        val result = PitchClassChordReranker.resolveEquivalentRoot("Am6", pitch, bass)

        assertTrue(result.changed)
        assertEquals("F#ø7", result.label)
    }

    @Test
    fun keepsAm6WhenAOwnsTheBass() {
        val pitch = evidence(
            "F#" to 0.91f,
            "A" to 1.00f,
            "C" to 0.92f,
            "E" to 0.88f,
        )
        val bass = evidence(
            "A" to 1.00f,
            "F#" to 0.45f,
            "E" to 0.20f,
        )

        val result = PitchClassChordReranker.resolveEquivalentRoot("Am6", pitch, bass)

        assertFalse(result.changed)
        assertEquals("Am6", result.label)
    }

    @Test
    fun sixNineWithRealInversionIsParsedAndPreserved() {
        val evidence = evidence(
            "G" to 1.00f,
            "B" to 0.92f,
            "D" to 0.87f,
            "E" to 0.84f,
            "A" to 0.81f,
        )

        val result = PitchClassChordReranker.rerank("G6/9/B", evidence)

        assertFalse(result.changed)
        assertEquals("G6/9/B", result.label)
    }

    private fun evidence(vararg values: Pair<String, Float>): FloatArray {
        val names = mapOf(
            "C" to 0, "C#" to 1, "Db" to 1, "D" to 2, "D#" to 3, "Eb" to 3,
            "E" to 4, "F" to 5, "F#" to 6, "Gb" to 6, "G" to 7,
            "G#" to 8, "Ab" to 8, "A" to 9, "A#" to 10, "Bb" to 10, "B" to 11,
        )
        val result = FloatArray(12)
        for ((name, value) in values) result[names.getValue(name)] = value
        return result
    }
}
