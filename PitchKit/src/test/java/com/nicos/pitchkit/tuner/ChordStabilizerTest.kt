package com.nicos.pitchkit.tuner

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ChordStabilizerTest {
    @Test
    fun switchesAfterAgreementEvenWhenNewChordScoreIsLower() {
        val stabilizer = ChordStabilizer()
        val c = candidate("C", 0.90, 0.10)
        val g = candidate("G", 0.70, 0.10)

        assertNull(stabilizer.accept(c))
        assertEquals("C", stabilizer.accept(c)?.name)
        assertEquals("C", stabilizer.accept(g)?.name)
        assertEquals("G", stabilizer.accept(g)?.name)
    }

    @Test
    fun ambiguousCandidateNeedsThreeFrameAgreement() {
        val stabilizer = ChordStabilizer()
        val c = candidate("C", 0.90, 0.10)
        val g = candidate("G", 0.70, 0.01)

        stabilizer.accept(c)
        assertEquals("C", stabilizer.accept(c)?.name)
        assertEquals("C", stabilizer.accept(g)?.name)
        assertEquals("C", stabilizer.accept(g)?.name)
        assertEquals("G", stabilizer.accept(g)?.name)
    }

    @Test
    fun sameChordCanBeReportedAgainAfterReset() {
        val stabilizer = ChordStabilizer()
        val c = candidate("C", 0.80, 0.10)

        stabilizer.accept(c)
        assertEquals("C", stabilizer.accept(c)?.name)
        stabilizer.reset()
        assertNull(stabilizer.accept(c))
        assertEquals("C", stabilizer.accept(c)?.name)
    }

    private fun candidate(name: String, score: Double, confidence: Double) = ChordDetector.ChordResult(
        name = name,
        score = score,
        confidence = confidence,
        bassPitchClass = 0,
    )
}
