package com.nicos.pitchkit.tuner.harmony.chordnet

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** Decodes the reusable plan-driven librosa CQT artifact format. */
object CqtPlanDecoder {
    private const val MAGIC = 0x5451434d
    private const val HEADER_BYTES = 128
    private const val OCTAVE_BYTES = 32

    fun decodeAndVerify(artifact: ByteArray): CqtPlan {
        require(artifact.size >= HEADER_BYTES) { "CQT plan is shorter than its header" }

        val data = ByteBuffer.wrap(artifact).order(ByteOrder.LITTLE_ENDIAN)
        require(data.getInt(0) == MAGIC) { "Invalid CQT plan magic" }
        require(data.getInt(8) == HEADER_BYTES) { "Unsupported CQT plan header size" }

        val formatVersion = data.getInt(4)
        val outputKind = data.getInt(12)
        require(outputKind in 0..1) { "Unsupported CQT output kind" }

        val config = CqtConfig(
            sampleRate = data.getDouble(16),
            hopLength = data.getInt(24),
            nBins = data.getInt(28),
            binsPerOctave = data.getInt(32),
            fmin = data.getDouble(40),
            logMagnitude = outputKind == 1,
        )

        val earlyDownsampleCount = data.getInt(36)
        val octaveCount = data.getInt(48)
        val tapCount = data.getInt(52)
        val coefficientCount = data.getInt(56)
        val payloadSize = data.getInt(60)
        require(HEADER_BYTES + payloadSize == artifact.size) { "Invalid CQT plan payload size" }

        val generatorLength = data.getInt(64)
        val generatorOffset = data.getInt(68)
        val octavesOffset = data.getInt(72)
        val rowOffsetsOffset = data.getInt(76)
        val fftBinsOffset = data.getInt(80)
        val coefficientsOffset = data.getInt(84)
        val binLengthsOffset = data.getInt(88)
        val downsampleOffset = data.getInt(92)

        val expectedPayloadSha = artifact
            .copyOfRange(96, 128)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        val actualPayloadSha = sha256(artifact.copyOfRange(HEADER_BYTES, artifact.size))
        require(actualPayloadSha == expectedPayloadSha) { "CQT plan payload SHA-256 mismatch" }

        val generatorBytes = artifact.copyOfRange(generatorOffset, generatorOffset + generatorLength)
        require(generatorBytes.all { (it.toInt() and 0xff) <= 0x7f }) { "CQT generator must be ASCII" }
        val generator = generatorBytes.toString(Charsets.US_ASCII)

        val octaves = List(octaveCount) { index ->
            val base = octavesOffset + index * OCTAVE_BYTES
            CqtOctavePlan(
                index = data.getInt(base),
                sampleRate = data.getDouble(base + 4),
                hopLength = data.getInt(base + 12),
                fftSize = data.getInt(base + 16),
                binStart = data.getInt(base + 20),
                binCount = data.getInt(base + 24),
            )
        }

        val rowOffsets = readIntArray(data, rowOffsetsOffset, config.nBins + 1)
        val fftBins = readIntArray(data, fftBinsOffset, coefficientCount)
        val coefficients = readFloatArray(data, coefficientsOffset, coefficientCount * 2)
        val binLengths = readFloatArray(data, binLengthsOffset, config.nBins)
        val halfCoefficientCount = (tapCount + 1) / 2
        val halfCoefficients = readFloatArray(data, downsampleOffset, halfCoefficientCount)

        require(config.sampleRate > 0.0)
        require(config.hopLength > 0)
        require(config.nBins > 0)
        require(config.binsPerOctave > 0)
        require(octaves.isNotEmpty())
        require(octaves.all { it.fftSize > 0 && it.fftSize and (it.fftSize - 1) == 0 }) {
            "CQT FFT sizes must be powers of two"
        }
        require(rowOffsets.firstOrNull() == 0)
        require(rowOffsets.lastOrNull() == coefficientCount)

        return CqtPlan(
            formatVersion = formatVersion,
            generator = generator,
            config = config,
            earlyDownsampleCount = earlyDownsampleCount,
            octaves = octaves,
            rowOffsets = rowOffsets,
            fftBins = fftBins,
            coefficients = coefficients,
            binLengths = binLengths,
            downsample = CqtDownsamplePlan(
                tapCount = tapCount,
                halfCoefficients = halfCoefficients,
            ),
            payloadSha256 = expectedPayloadSha,
        )
    }

    private fun readIntArray(buffer: ByteBuffer, offset: Int, count: Int): IntArray =
        IntArray(count) { index -> buffer.getInt(offset + index * Int.SIZE_BYTES) }

    private fun readFloatArray(buffer: ByteBuffer, offset: Int, count: Int): FloatArray =
        FloatArray(count) { index -> buffer.getFloat(offset + index * Float.SIZE_BYTES) }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
