package com.nicos.pitchkit.tuner.models

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class AudioFrameTest {
    @Test
    fun `mono frame is returned unchanged`() {
        val samples = floatArrayOf(-1f, -0.5f, 0f, 0.5f, 1f)
        val frame = AudioFrame(samples, sampleRate = 48000)

        assertArrayEquals(samples, frame.toMono(), 0.0001f)
    }

    @Test
    fun `stereo interleaved frame is downmixed to mono`() {
        val frame = AudioFrame(
            samples = floatArrayOf(
                1f, -1f,
                0.5f, 0.5f,
                -0.25f, 0.75f,
            ),
            sampleRate = 48000,
            channelCount = 2,
        )

        assertArrayEquals(
            floatArrayOf(0f, 0.5f, 0.25f),
            frame.toMono(),
            0.0001f,
        )
    }
}
