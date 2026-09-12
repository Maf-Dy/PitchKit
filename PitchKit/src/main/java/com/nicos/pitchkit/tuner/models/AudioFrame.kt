package com.nicos.pitchkit.tuner.models

/**
 * Source-agnostic PCM frame consumed by the analyzer.
 *
 * Samples are normalized floating-point PCM in the -1f..1f range. Multi-channel
 * frames are interleaved (L, R, L, R...) and are downmixed to mono by the core.
 */
class AudioFrame(
    val samples: FloatArray,
    val sampleRate: Int,
    val channelCount: Int = 1,
) {
    init {
        require(sampleRate > 0) { "sampleRate must be > 0" }
        require(channelCount > 0) { "channelCount must be > 0" }
        require(samples.size % channelCount == 0) {
            "Interleaved sample count must be divisible by channelCount"
        }
    }

    internal fun toMono(): FloatArray {
        if (channelCount == 1) return samples

        val frameCount = samples.size / channelCount
        return FloatArray(frameCount) { frameIndex ->
            var sum = 0f
            val base = frameIndex * channelCount
            for (channel in 0 until channelCount) {
                sum += samples[base + channel]
            }
            sum / channelCount
        }
    }
}
