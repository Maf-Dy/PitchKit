package com.nicos.pitchkit.tuner.harmony.crema

import org.junit.Assert.assertEquals
import org.junit.Test

class CremaSongViterbiDecoderTest {
    @Test
    fun usesOneBassDecisionAcrossDecodedChordSegment() {
        val state = CremaRuntimeState(
            labels = listOf("N", "X", "D:hdim7"),
            transitionDiagonal = 0.95,
            transitionOffDiagonal = 0.01,
        )
        val decoder = CremaSongViterbiDecoder(state, preferFlats = false)
        val heads = CremaHeads(
            frames = 2,
            tag = FloatArray(2 * CremaContract.CHORD_COUNT).also {
                it[2] = 0.92f
                it[CremaContract.CHORD_COUNT + 2] = 0.91f
            },
            pitch = FloatArray(2 * CremaContract.PITCH_COUNT),
            root = FloatArray(2 * CremaContract.ROOT_COUNT),
            bass = FloatArray(2 * CremaContract.BASS_COUNT).also {
                // Both frames support C as the segment bass, but the first frame
                // alone would have been close enough to fluctuate toward D.
                it[0] = 0.80f
                it[2] = 0.70f
                val row1 = CremaContract.BASS_COUNT
                it[row1 + 0] = 0.90f
                it[row1 + 2] = 0.40f
            },
        )

        decoder.add(heads, localFrame = 0, globalFrame = 0)
        decoder.add(heads, localFrame = 1, globalFrame = 1)
        val result = decoder.decode()

        assertEquals(2, result.size)
        assertEquals("Dø7/C", result[0].label)
        assertEquals("Dø7/C", result[1].label)
    }
}
