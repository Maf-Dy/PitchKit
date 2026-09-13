package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class ChordDetectorTest {
    private val sampleRate = 44_100
    private val detector = ChordDetector(sampleRate, InstrumentProfile.Guitar)

    @Test
    fun detectsCommonOpenChordSpectra() {
        val cases = mapOf(
            "C" to doubleArrayOf(130.81, 164.81, 196.00, 261.63, 329.63, 392.00),
            "G" to doubleArrayOf(98.00, 123.47, 146.83, 196.00, 246.94, 392.00),
            "Am" to doubleArrayOf(110.00, 164.81, 220.00, 261.63, 329.63),
            "F" to doubleArrayOf(87.31, 130.81, 174.61, 220.00, 261.63, 349.23),
            "Em" to doubleArrayOf(82.41, 123.47, 164.81, 196.00, 246.94, 329.63),
            "Dm" to doubleArrayOf(146.83, 220.00, 293.66, 349.23, 440.00),
        )

        assertChordCases(cases)
    }

    @Test
    fun detectsSeventhChordsInsteadOfTheirTriadSubsets() {
        val cases = mapOf(
            "A7" to doubleArrayOf(220.00, 277.18, 329.63, 392.00),
            "Cmaj7" to doubleArrayOf(261.63, 329.63, 392.00, 493.88),
            "C7" to doubleArrayOf(261.63, 329.63, 392.00, 466.16),
            "D7" to doubleArrayOf(293.66, 369.99, 440.00, 523.25),
        )

        assertChordCases(cases)
    }

    @Test
    fun recognizesFullyDiminishedSeventhVoicing() {
        // G-Bb-Db-E: all four notes are separated by minor thirds.
        // This used to be forced through a diminished-triad template and could
        // compete with unrelated dominant-seventh candidates.
        val result = detector.detect(mix(doubleArrayOf(196.00, 233.08, 277.18, 329.63)))
        assertNotNull(result)
        assertEquals("Gdim7", result?.name)
    }

    private fun assertChordCases(cases: Map<String, DoubleArray>) {
        for ((expected, frequencies) in cases) {
            val result = detector.detect(mix(frequencies))
            assertNotNull("No result for $expected", result)
            assertEquals(expected, result?.name)
        }
    }

    private fun mix(frequencies: DoubleArray): FloatArray = FloatArray(8192) { index ->
        (frequencies.sumOf { frequency ->
            sin(2.0 * PI * frequency * index / sampleRate)
        } / frequencies.size).toFloat()
    }
}
