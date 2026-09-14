package com.nicos.pitchkit.tuner.harmony.consonance

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ConsonanceChordDecoderTest {
    private val decoder = ConsonanceChordDecoder()

    @Test
    fun decodesG6Over9FromDecomposedPitches() {
        val result = decoder.decode(heads(root = 7, bass = 7, pitches = intArrayOf(7, 11, 2, 4, 9))).single()
        assertEquals("G6/9", result.label)
        assertEquals("G", result.root)
        assertEquals("G", result.bass)
        assertTrue(result.pitchClasses.containsAll(listOf("G", "B", "D", "E", "A")))
    }

    @Test
    fun decodesHalfDiminishedFromDecomposedPitches() {
        val result = decoder.decode(heads(root = 6, bass = 6, pitches = intArrayOf(6, 9, 0, 4))).single()
        assertEquals("F#ø7", result.label)
        assertEquals("F#", result.root)
    }

    @Test
    fun decodesFullyDiminishedSeventh() {
        val result = decoder.decode(heads(root = 6, bass = 6, pitches = intArrayOf(6, 9, 0, 3))).single()
        assertEquals("F#dim7", result.label)
    }

    private fun heads(root: Int, bass: Int, pitches: IntArray): ConsonanceHeads {
        val rootLogits = FloatArray(13) { -6f }.also { it[root] = 6f }
        val bassLogits = FloatArray(13) { -6f }.also { it[bass] = 6f }
        val pitchLogits = FloatArray(12) { -6f }
        for (pc in pitches) pitchLogits[pc] = 6f
        return ConsonanceHeads(
            frames = 1,
            root = rootLogits,
            bass = bassLogits,
            pitch = pitchLogits,
        )
    }
}
