package com.nicos.pitchkit.tuner.harmony.crema

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class CremaHarmonyDecoderTest {
    @Test
    fun decodesDHalfDiminishedFromChordTag() {
        val state = CremaRuntimeState(
            labels = listOf("N", "X", "D:hdim7"),
            transitionDiagonal = 0.95,
            transitionOffDiagonal = 0.01,
        )
        val heads = CremaHeads(
            frames = 1,
            tag = FloatArray(CremaContract.CHORD_COUNT).also { it[2] = 0.92f },
            pitch = FloatArray(CremaContract.PITCH_COUNT).also {
                it[2] = 0.95f
                it[5] = 0.93f
                it[8] = 0.91f
                it[0] = 0.94f
            },
            root = FloatArray(CremaContract.ROOT_COUNT).also { it[2] = 0.97f },
            bass = FloatArray(CremaContract.BASS_COUNT).also { it[2] = 0.96f },
        )

        val result = CremaHarmonyDecoder(state).decode(heads, smoothingFrames = 1)

        assertNotNull(result)
        assertEquals("Dø7", result!!.label)
        assertEquals("D:hdim7", result.rawLabel)
    }

    @Test
    fun auxiliaryHeadsDoNotOverrideClearChordTag() {
        val state = CremaRuntimeState(
            labels = listOf("N", "X", "C:maj", "A:min"),
            transitionDiagonal = 0.95,
            transitionOffDiagonal = 0.01,
        )
        val heads = CremaHeads(
            frames = 1,
            tag = FloatArray(CremaContract.CHORD_COUNT).also {
                it[2] = 0.87f
                it[3] = 0.31f
            },
            // Deliberately make the structured heads look strongly like A minor.
            // The reference Crema chord class is still the chord-tag prediction.
            pitch = FloatArray(CremaContract.PITCH_COUNT).also {
                it[9] = 1.00f
                it[0] = 0.98f
                it[4] = 0.96f
            },
            root = FloatArray(CremaContract.ROOT_COUNT).also {
                it[9] = 0.99f
                it[0] = 0.05f
            },
            bass = FloatArray(CremaContract.BASS_COUNT).also { it[0] = 0.95f },
        )

        val result = CremaHarmonyDecoder(state).decode(heads, smoothingFrames = 1)

        assertEquals("C", result!!.label)
        assertEquals("C:maj", result.rawLabel)
        assertEquals(0.87, result.confidence, 1e-6)
    }

    @Test
    fun preservesValidHalfDiminishedInversion() {
        val state = CremaRuntimeState(
            labels = listOf("N", "X", "D:hdim7"),
            transitionDiagonal = 0.95,
            transitionOffDiagonal = 0.01,
        )
        val heads = CremaHeads(
            frames = 1,
            tag = FloatArray(CremaContract.CHORD_COUNT).also { it[2] = 0.92f },
            pitch = FloatArray(CremaContract.PITCH_COUNT).also {
                it[2] = 0.95f
                it[5] = 0.93f
                it[8] = 0.91f
                it[0] = 0.94f
            },
            root = FloatArray(CremaContract.ROOT_COUNT).also { it[2] = 0.97f },
            bass = FloatArray(CremaContract.BASS_COUNT).also { it[0] = 0.96f },
        )

        val result = CremaHarmonyDecoder(state).decode(heads, smoothingFrames = 1)

        assertEquals("Dø7/C", result!!.label)
    }
}
