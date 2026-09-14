package com.nicos.pitchkit.tuner.harmony.chordnet

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LivePitchEvidenceAccumulatorTest {
    @Test
    fun arpeggioWaitsUntilPitchSetStopsGrowing() {
        val accumulator = LivePitchEvidenceAccumulator()

        var snapshot = accumulator.update(evidence(6), evidence(6)) // F#
        assertFalse(snapshot.ready)

        snapshot = accumulator.update(evidence(6, 9), evidence(6)) // + A
        assertFalse(snapshot.ready)

        snapshot = accumulator.update(evidence(6, 9, 0), evidence(6)) // + C
        assertFalse(snapshot.ready)

        snapshot = accumulator.update(evidence(6, 9, 0, 4), evidence(6)) // + E
        assertFalse(snapshot.ready)

        snapshot = accumulator.update(evidence(6, 9, 0, 4), evidence(6))
        assertTrue(snapshot.ready)
        assertTrue(snapshot.pitch[6] > 0.2f)
        assertTrue(snapshot.pitch[9] > 0.2f)
        assertTrue(snapshot.pitch[0] > 0.2f)
        assertTrue(snapshot.pitch[4] > 0.2f)
    }

    @Test
    fun simultaneousChordCanResolveAfterOneStableUpdate() {
        val accumulator = LivePitchEvidenceAccumulator()
        val chord = evidence(7, 11, 2, 5) // G7

        assertFalse(accumulator.update(chord, evidence(7)).ready)
        assertTrue(accumulator.update(chord, evidence(7)).ready)
    }

    @Test
    fun newPitchAfterReadyMakesGestureFormAgain() {
        val accumulator = LivePitchEvidenceAccumulator()
        val g = evidence(7, 11, 2)

        assertFalse(accumulator.update(g, evidence(7)).ready)
        assertTrue(accumulator.update(g, evidence(7)).ready)

        val g7 = evidence(7, 11, 2, 5)
        assertFalse(accumulator.update(g7, evidence(7)).ready)
        assertTrue(accumulator.update(g7, evidence(7)).ready)
    }

    private fun evidence(vararg pitchClasses: Int): FloatArray = FloatArray(12) { index ->
        if (index in pitchClasses) 1f else 0f
    }
}
