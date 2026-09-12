package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.InstrumentProfile
import org.junit.Assert.assertNotNull
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class ChordDetectorTest {
    @Test
    fun `major triad produces a chord candidate`() {
        val sampleRate = 44100
        val detector = ChordDetector(
            sampleRate = sampleRate,
            profile = InstrumentProfile.Guitar,
        )

        val frame = triad(
            sampleRate = sampleRate,
            sampleCount = 8192,
            frequencies = doubleArrayOf(261.63, 329.63, 392.00), // C E G
        )

        // Temporal chroma uses several frames; feed the same sustained chord.
        var result: ChordDetector.ChordResult? = null
        repeat(4) {
            result = detector.detect(frame, minScore = 0.10)
        }

        assertNotNull(result)
    }

    private fun triad(
        sampleRate: Int,
        sampleCount: Int,
        frequencies: DoubleArray,
    ): FloatArray {
        return FloatArray(sampleCount) { index ->
            val time = index.toDouble() / sampleRate
            (frequencies.sumOf { frequency ->
                sin(2.0 * PI * frequency * time)
            } / frequencies.size).toFloat()
        }
    }
}
