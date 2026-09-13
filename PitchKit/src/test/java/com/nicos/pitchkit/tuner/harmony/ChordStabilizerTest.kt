package com.nicos.pitchkit.tuner.harmony

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ChordStabilizerTest {
    private fun chord(label: String, confidence: Double = 0.8) =
        ChordRecognition(label = label, confidence = confidence, backend = "test")

    @Test
    fun firstPredictionPublishesImmediately() {
        val stabilizer = ChordStabilizer(minimumConfidence = 0.10)

        assertEquals("C", stabilizer.update(chord("C"))?.label)
        assertEquals("C", stabilizer.currentWithoutPrediction()?.label)
    }

    @Test
    fun differentChordSuppressesOldUntilConfirmed() {
        val stabilizer = ChordStabilizer(
            minimumConfidence = 0.10,
            changeConfirmations = 2,
        )

        assertEquals("C", stabilizer.update(chord("C"))?.label)

        assertNull(stabilizer.update(chord("G")))
        assertNull(stabilizer.currentWithoutPrediction())

        assertEquals("G", stabilizer.update(chord("G"))?.label)
        assertEquals("G", stabilizer.currentWithoutPrediction()?.label)
    }

    @Test
    fun weakPredictionDoesNotLeakPreviousChordAsLiveResult() {
        val stabilizer = ChordStabilizer(minimumConfidence = 0.20)

        assertEquals("Am", stabilizer.update(chord("Am", 0.90))?.label)
        assertNull(stabilizer.update(chord("F", 0.05)))
        assertNull(stabilizer.currentWithoutPrediction())

        assertEquals("Am", stabilizer.update(chord("Am", 0.70))?.label)
    }

    @Test
    fun resetRemovesAllStableAndPendingState() {
        val stabilizer = ChordStabilizer(minimumConfidence = 0.10)
        stabilizer.update(chord("C"))
        stabilizer.update(chord("G"))

        stabilizer.reset()

        assertNull(stabilizer.currentWithoutPrediction())
        assertEquals("G", stabilizer.update(chord("G"))?.label)
    }
}
