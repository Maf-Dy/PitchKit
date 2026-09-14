package com.nicos.pitchkit.tuner.harmony.chordnet

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class CqtChordTemplateDetectorTest {
    @Test
    fun detectsHalfDiminishedWhenFSharpIsBass() {
        val result = CqtChordTemplateDetector.detect(
            pitchEvidence = evidence(
                "F#" to 1.00f,
                "A" to 0.93f,
                "C" to 0.89f,
                "E" to 0.86f,
            ),
            bassEvidence = evidence(
                "F#" to 1.00f,
                "A" to 0.20f,
            ),
        )

        assertNotNull(result)
        assertEquals("F#ø7", result!!.label)
    }

    @Test
    fun detectsAMinorSixForSamePitchSetWhenAIsBass() {
        val result = CqtChordTemplateDetector.detect(
            pitchEvidence = evidence(
                "F#" to 1.00f,
                "A" to 0.93f,
                "C" to 0.89f,
                "E" to 0.86f,
            ),
            bassEvidence = evidence(
                "A" to 1.00f,
                "F#" to 0.20f,
            ),
        )

        assertNotNull(result)
        assertEquals("Am6", result!!.label)
    }

    @Test
    fun detectsFullyDiminishedSevenUsingBassToChooseSymmetricRoot() {
        val result = CqtChordTemplateDetector.detect(
            pitchEvidence = evidence(
                "F#" to 1.00f,
                "A" to 0.93f,
                "C" to 0.89f,
                "D#" to 0.86f,
            ),
            bassEvidence = evidence(
                "F#" to 1.00f,
                "A" to 0.20f,
                "C" to 0.15f,
                "D#" to 0.10f,
            ),
        )

        assertNotNull(result)
        assertEquals("F#dim7", result!!.label)
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
