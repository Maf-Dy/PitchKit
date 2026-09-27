package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.AudioFrame
import com.nicos.pitchkit.tuner.models.InstrumentProfile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sin

class PitchAnalyzerAccuracyTest {
    private val sampleRate = 48_000
    private val size = 4_096

    private fun cents(detected: Float, expected: Double) = 1200.0 * ln(detected / expected) / ln(2.0)

    private fun sine(frequency: Double, phase: Double, noise: Double = 0.0, seed: Long = 1): FloatArray {
        val random = Random(seed)
        return FloatArray(size) { index ->
            (0.3 * sin(2.0 * PI * frequency * index / sampleRate + phase) + noise * random.nextGaussian()).toFloat()
        }
    }

    private fun note(profile: InstrumentProfile, samples: FloatArray): TuningResult.Note {
        val result = PitchAnalyzer(profile, DetectionMode.NOTE).process(AudioFrame(samples, sampleRate))
        assertTrue("Expected a note, got $result", result is TuningResult.Note)
        return result as TuningResult.Note
    }

    // The per-window high-pass used to pull these by 3 (E2) to 20 (E1) cents depending on phase.
    @Test
    fun lowStringsReadWithinHalfACentAtAnyPhase() {
        listOf(InstrumentProfile.Bass to 41.20, InstrumentProfile.Bass to 55.0, InstrumentProfile.Guitar to 82.41)
            .forEach { (profile, frequency) ->
                repeat(8) { step ->
                    val result = note(profile, sine(frequency, phase = step * PI / 4))
                    val error = cents(result.freq, frequency)
                    assertTrue("$frequency Hz at phase $step read $error cents", abs(error) < 0.5)
                }
            }
    }

    @Test
    fun highNotesReadWithinHalfACent() {
        listOf(987.77, 1318.51).forEach { frequency ->
            val result = note(InstrumentProfile.Guitar, sine(frequency, phase = 0.3))
            assertTrue("$frequency Hz read ${cents(result.freq, frequency)} cents", abs(cents(result.freq, frequency)) < 0.5)
        }
    }

    @Test
    fun reportsTheOctave() {
        assertEquals("E2", note(InstrumentProfile.Guitar, sine(82.41, phase = 0.0)).nameWithOctave)
        assertEquals("A1", note(InstrumentProfile.Bass, sine(55.0, phase = 0.0)).nameWithOctave)
    }

    // At this noise level no lag clears the 0.15 threshold; the global-minimum fallback still finds the note.
    @Test
    fun noisyInputStillProducesTheRightNote() {
        val result = note(InstrumentProfile.Guitar, sine(196.0, phase = 0.0, noise = 0.3 / Math.sqrt(2.0) / 1.8))
        assertEquals("G3", result.nameWithOctave)
    }

    @Test
    fun pureNoiseIsNotANote() {
        val random = Random(9)
        val noise = FloatArray(size) { (0.1 * random.nextGaussian()).toFloat() }
        val result = PitchAnalyzer(InstrumentProfile.Guitar, DetectionMode.NOTE).process(AudioFrame(noise, sampleRate))
        assertEquals(TuningResult.Silence, result)
    }
}
