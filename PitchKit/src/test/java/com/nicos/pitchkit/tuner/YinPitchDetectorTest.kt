package com.nicos.pitchkit.tuner

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class YinPitchDetectorTest {
    private val sampleRate = 44_100

    @Test
    fun detectsLowGuitarEWithBoundedLagSearch() {
        val detector = YinPitchDetector(
            sampleRate = sampleRate,
            minFrequencyHz = 70.0,
            maxFrequencyHz = 5_000.0,
        )

        val detected = detector.detect(sineWave(frequency = 82.41, size = 4_096))

        assertEquals(82.41, detected.toDouble(), 0.6)
    }

    @Test
    fun detectsA440WithBoundedLagSearch() {
        val detector = YinPitchDetector(
            sampleRate = sampleRate,
            minFrequencyHz = 70.0,
            maxFrequencyHz = 5_000.0,
        )

        val detected = detector.detect(sineWave(frequency = 440.0, size = 4_096))

        assertEquals(440.0, detected.toDouble(), 0.8)
    }

    private fun sineWave(frequency: Double, size: Int): FloatArray =
        FloatArray(size) { index ->
            (0.6 * sin(2.0 * PI * frequency * index / sampleRate)).toFloat()
        }
}
