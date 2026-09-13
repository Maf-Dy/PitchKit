package com.nicos.pitchkit.tuner.models

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class AudioFrameTest {
    @Test
    fun `stereo PCM is averaged to mono`() {
        val frame = AudioFrame(
            samples = floatArrayOf(1f, -1f, 0.5f, 0.5f),
            sampleRate = 48000,
            channelCount = 2,
        )

        assertArrayEquals(floatArrayOf(0f, 0.5f), frame.toMono(), 0.0001f)
    }
}
