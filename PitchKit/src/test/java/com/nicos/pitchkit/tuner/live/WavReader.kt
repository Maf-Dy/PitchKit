package com.nicos.pitchkit.tuner.live

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Decoded PCM from a RIFF/WAVE file.
 *
 * [samples] is interleaved and normalised to roughly -1..1, exactly the shape
 * [com.nicos.pitchkit.tuner.models.AudioFrame] expects.
 */
class WavAudio(
    val samples: FloatArray,
    val sampleRate: Int,
    val channelCount: Int,
) {
    val frameCount: Int get() = samples.size / channelCount

    val durationSeconds: Double get() = frameCount.toDouble() / sampleRate

    /** Channel average, matching `AudioFrame.toMono()`. */
    fun toMono(): FloatArray {
        if (channelCount == 1) return samples
        return FloatArray(frameCount) { frame ->
            var sum = 0f
            val base = frame * channelCount
            for (channel in 0 until channelCount) sum += samples[base + channel]
            sum / channelCount
        }
    }
}

/**
 * Dependency-free WAV reader for the live replay harness.
 *
 * Supports the two subtypes the corpus renderers produce — 16-bit signed PCM and
 * 32-bit IEEE float — plus 24/32-bit integer PCM, in plain `WAVE_FORMAT_PCM`,
 * `WAVE_FORMAT_IEEE_FLOAT` and `WAVE_FORMAT_EXTENSIBLE` containers. It reads the
 * whole file into memory: corpus items are seconds long, not songs.
 */
object WavReader {
    private const val FORMAT_PCM = 1
    private const val FORMAT_FLOAT = 3
    private const val FORMAT_EXTENSIBLE = 0xFFFE

    fun read(path: String): WavAudio = read(File(path))

    fun read(file: File): WavAudio {
        require(file.isFile) { "WAV file not found: ${file.absolutePath}" }
        val buffer = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        require(buffer.remaining() >= 12) { "${file.name}: too short to be a RIFF file" }
        require(tag(buffer) == "RIFF") { "${file.name}: not a RIFF file" }
        buffer.int // RIFF size; trusted only as far as the actual file length.
        require(tag(buffer) == "WAVE") { "${file.name}: not a WAVE file" }

        var formatCode = -1
        var channelCount = 0
        var sampleRate = 0
        var bitsPerSample = 0
        var data: ByteArray? = null

        while (buffer.remaining() >= 8) {
            val id = tag(buffer)
            val size = buffer.int
            require(size >= 0 && size <= buffer.remaining()) {
                "${file.name}: chunk '$id' declares $size bytes, ${buffer.remaining()} remain"
            }
            val next = buffer.position() + size + (size and 1)
            when (id) {
                "fmt " -> {
                    require(size >= 16) { "${file.name}: fmt chunk is $size bytes" }
                    formatCode = buffer.short.toInt() and 0xFFFF
                    channelCount = buffer.short.toInt() and 0xFFFF
                    sampleRate = buffer.int
                    buffer.int // byte rate
                    buffer.short // block align
                    bitsPerSample = buffer.short.toInt() and 0xFFFF
                    if (formatCode == FORMAT_EXTENSIBLE && size >= 40) {
                        buffer.short // cbSize
                        buffer.short // valid bits
                        buffer.int // channel mask
                        formatCode = buffer.short.toInt() and 0xFFFF // first field of the subformat GUID
                    }
                }

                "data" -> {
                    val bytes = ByteArray(size)
                    buffer.get(bytes)
                    data = bytes
                }
            }
            buffer.position(next.coerceAtMost(buffer.limit()))
        }

        val payload = requireNotNull(data) { "${file.name}: no data chunk" }
        require(channelCount > 0 && sampleRate > 0) { "${file.name}: no usable fmt chunk" }
        val samples = decode(file.name, payload, formatCode, bitsPerSample)
        require(samples.size % channelCount == 0) {
            "${file.name}: ${samples.size} samples is not divisible by $channelCount channels"
        }
        return WavAudio(samples, sampleRate, channelCount)
    }

    private fun decode(name: String, payload: ByteArray, formatCode: Int, bits: Int): FloatArray {
        val source = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN)
        return when {
            formatCode == FORMAT_FLOAT && bits == 32 ->
                FloatArray(payload.size / 4) { source.float }

            formatCode == FORMAT_FLOAT && bits == 64 ->
                FloatArray(payload.size / 8) { source.double.toFloat() }

            formatCode == FORMAT_PCM && bits == 16 ->
                FloatArray(payload.size / 2) { source.short / 32768f }

            formatCode == FORMAT_PCM && bits == 8 ->
                FloatArray(payload.size) { ((payload[it].toInt() and 0xFF) - 128) / 128f }

            formatCode == FORMAT_PCM && bits == 24 -> FloatArray(payload.size / 3) { index ->
                val base = index * 3
                val value = (payload[base].toInt() and 0xFF) or
                    ((payload[base + 1].toInt() and 0xFF) shl 8) or
                    (payload[base + 2].toInt() shl 16)
                value / 8_388_608f
            }

            formatCode == FORMAT_PCM && bits == 32 ->
                FloatArray(payload.size / 4) { source.int / 2_147_483_648f }

            else -> error("$name: unsupported WAV format code $formatCode at $bits bits")
        }
    }

    private fun tag(buffer: ByteBuffer): String {
        val bytes = ByteArray(4)
        buffer.get(bytes)
        return String(bytes, Charsets.US_ASCII)
    }
}
