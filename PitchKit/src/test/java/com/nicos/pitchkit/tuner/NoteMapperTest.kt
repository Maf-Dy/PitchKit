package com.nicos.pitchkit.tuner

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NoteMapperTest {
    @Test
    fun `440 Hz is A at A440`() {
        val result = requireNotNull(NoteMapper.frequencyToNote(440f, referenceA4Hz = 440.0))
        assertEquals("A", result.name)
        assertTrue(kotlin.math.abs(result.cents) < 0.01)
    }

    @Test
    fun `432 Hz is A at A432`() {
        val result = requireNotNull(NoteMapper.frequencyToNote(432f, referenceA4Hz = 432.0))
        assertEquals("A", result.name)
        assertTrue(kotlin.math.abs(result.cents) < 0.01)
    }

    @Test
    fun `432 Hz is flat when reference remains A440`() {
        val result = requireNotNull(NoteMapper.frequencyToNote(432f, referenceA4Hz = 440.0))
        assertEquals("A", result.name)
        assertTrue(result.cents < -30.0)
    }
}
