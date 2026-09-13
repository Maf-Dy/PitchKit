package com.nicos.pitchkit.tuner

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import org.junit.Assert.assertTrue
import org.junit.Test

class YinPitchDetectorTest {
    @Test
    fun detectsA4WithinOneHertz() {
        val sampleRate = 44_100
        val samples = FloatArray(4096) { index ->
            sin(2.0 * PI * 440.0 * index / sampleRate).toFloat()
        }
        val detector = YinPitchDetector(sampleRate, 70.0, 1200.0)
        val detected = detector.detect(samples)
        assertTrue("Expected ~440 Hz, got $detected", abs(detected - 440f) < 1f)
    }
}
