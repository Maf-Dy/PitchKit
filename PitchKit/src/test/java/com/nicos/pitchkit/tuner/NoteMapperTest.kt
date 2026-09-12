package com.nicos.pitchkit.tuner

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NoteMapperTest {
    @Test
    fun `A4 is in tune at 440 when reference is 440`() {
        val result = requireNotNull(
            NoteMapper.frequencyToNote(
                freq = 440f,
                referenceA4Hz = 440.0,
            )
        )

        assertEquals("A", result.name)
        assertEquals("A4", result.nameWithOctave)
        assertEquals(0.0, result.cents, 0.001)
    }

    @Test
    fun `A4 is in tune at 432 when reference is 432`() {
        val result = requireNotNull(
            NoteMapper.frequencyToNote(
                freq = 432f,
                referenceA4Hz = 432.0,
            )
        )

        assertEquals("A", result.name)
        assertEquals("A4", result.nameWithOctave)
        assertEquals(0.0, result.cents, 0.001)
    }

    @Test
    fun `440 is sharp when reference is 432`() {
        val result = requireNotNull(
            NoteMapper.frequencyToNote(
                freq = 440f,
                referenceA4Hz = 432.0,
            )
        )

        assertEquals("A", result.name)
        assertTrue(result.cents in 31.0..33.0)
    }
}
