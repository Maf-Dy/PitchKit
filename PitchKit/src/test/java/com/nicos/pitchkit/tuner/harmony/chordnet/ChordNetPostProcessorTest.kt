package com.nicos.pitchkit.tuner.harmony.chordnet

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChordNetPostProcessorTest {
    @Test
    fun exposesTopThreeCandidatesInScoreOrder() {
        val logits = FloatArray(
            ChordNetContract.SEQUENCE_LENGTH * ChordNetContract.CHORD_COUNT
        ) { -10f }

        val rootOffset = 6 * 14 // F#
        val min7 = rootOffset + 6
        val dim7 = rootOffset + 10
        val hdim7 = rootOffset + 11

        logits[hdim7] = 6f
        logits[dim7] = 5f
        logits[min7] = 4f

        val prediction = ChordNetPostProcessor.decode(
            logits = logits,
            windowCount = 1,
            validFrameCount = 1,
            smoothingKernel = 1,
            includeAlternatives = true,
        ).single()

        assertEquals("F#:hdim7", prediction.rawLabel)
        assertEquals("F#ø7", prediction.displayLabel)
        assertEquals(
            listOf("F#:hdim7", "F#:dim7", "F#:min7"),
            prediction.alternatives.map { it.rawLabel },
        )
        assertTrue(prediction.alternatives[0].confidence > prediction.alternatives[1].confidence)
        assertTrue(prediction.alternatives[1].confidence > prediction.alternatives[2].confidence)
    }
}
