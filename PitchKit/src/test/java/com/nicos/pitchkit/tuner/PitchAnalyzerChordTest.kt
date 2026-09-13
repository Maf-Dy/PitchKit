package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.AudioFrame
import com.nicos.pitchkit.tuner.models.InstrumentProfile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin

class PitchAnalyzerChordTest {
    private val sampleRate = 44_100
    private val frameSize = 8_192

    @Test
    fun detectsCommonGuitarTriadsWithHarmonics() {
        val cases = listOf(
            "C" to intArrayOf(48, 52, 55, 60, 64),
            "G" to intArrayOf(43, 47, 50, 55, 59, 67),
            "Am" to intArrayOf(45, 52, 57, 60, 64),
            "Em" to intArrayOf(40, 47, 52, 55, 59, 64),
        )

        for ((expectedChord, midiNotes) in cases) {
            val analyzer = PitchAnalyzer(
                profile = InstrumentProfile.Guitar,
                mode = DetectionMode.CHORD,
                referenceA4Hz = 440.0,
                chordMinScore = 0.20,
            )

            val frame = AudioFrame(
                samples = synthesizeGuitarChord(midiNotes),
                sampleRate = sampleRate,
            )

            analyzer.process(frame)
            val result = analyzer.process(frame)

            assertTrue("Expected a chord for $expectedChord but got $result", result is TuningResult.Chord)
            assertEquals(expectedChord, (result as TuningResult.Chord).name)
        }
    }

    private fun synthesizeGuitarChord(midiNotes: IntArray): FloatArray {
        val output = FloatArray(frameSize)

        for ((noteIndex, midi) in midiNotes.withIndex()) {
            val fundamental = 440.0 * 2.0.pow((midi - 69) / 12.0)
            for (harmonic in 1..6) {
                val amplitude = 0.17 / harmonic
                val phase = noteIndex * 0.37 * harmonic
                for (sampleIndex in output.indices) {
                    val time = sampleIndex.toDouble() / sampleRate
                    output[sampleIndex] += (
                        amplitude * sin(2.0 * PI * fundamental * harmonic * time + phase)
                    ).toFloat()
                }
            }
        }

        for (sampleIndex in output.indices) {
            val time = sampleIndex.toDouble() / sampleRate
            val attack = (time / 0.01).coerceIn(0.0, 1.0)
            val decay = exp(-2.0 * time)
            output[sampleIndex] = (output[sampleIndex] * attack * decay)
                .coerceIn(-0.95, 0.95)
                .toFloat()
        }

        return output
    }
}
