package com.nicos.pitchkit.tuner.harmony.chordnet

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ChordNetContractTest {
    @Test
    fun vocabularyMatchesExportOrdering() {
        assertEquals(170, ChordNetVocabulary.labels.size)
        assertEquals("C:min", ChordNetVocabulary.labelAt(0))
        assertEquals("C", ChordNetVocabulary.labelAt(1))
        assertEquals("C:sus4", ChordNetVocabulary.labelAt(13))
        assertEquals("C#:min", ChordNetVocabulary.labelAt(14))
        assertEquals("X", ChordNetVocabulary.labelAt(168))
        assertEquals("N", ChordNetVocabulary.labelAt(169))
        assertEquals("Am7", ChordNetVocabulary.toDisplay("A:min7"))
        assertNull(ChordNetVocabulary.displayLabel(169))
    }

    @Test
    fun postProcessorSmoothsAndDecodesExpectedChord() {
        val frames = 12
        val logits = FloatArray(ChordNetContract.SEQUENCE_LENGTH * ChordNetContract.CHORD_COUNT) {
            -5f
        }
        val cMajorIndex = 1
        for (frame in 0 until frames) {
            logits[frame * ChordNetContract.CHORD_COUNT + cMajorIndex] = 6f
        }

        val decoded = ChordNetPostProcessor.decode(
            logits = logits,
            windowCount = 1,
            validFrameCount = frames,
        )

        assertEquals(frames, decoded.size)
        assertTrue(decoded.all { it.displayLabel == "C" })
        assertTrue(decoded.all { it.confidence > 0.9 })
    }
}
