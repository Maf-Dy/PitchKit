package com.nicos.pitchkit.tuner.harmony.lvchordia

import com.nicos.pitchkit.tuner.FFT
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sqrt

internal data class LvChordiaPseudoPlan(
    val sampleRate: Double,
    val hopLength: Int,
    val fftSize: Int,
    val binCount: Int,
    val rowOffsets: IntArray,
    val fftBins: IntArray,
    val coefficients: FloatArray,
)

internal object LvChordiaPseudoPlanDecoder {
    private const val MAGIC = 0x3150564c
    private const val HEADER_BYTES = 96

    fun decodeAndVerify(artifact: ByteArray): LvChordiaPseudoPlan {
        require(artifact.size >= HEADER_BYTES) { "LV-Chordia pseudo-CQT plan is truncated" }
        val data = ByteBuffer.wrap(artifact).order(ByteOrder.LITTLE_ENDIAN)
        require(data.getInt(0) == MAGIC) { "Invalid LV-Chordia pseudo-CQT magic" }
        require(data.getInt(4) == 1) { "Unsupported LV-Chordia pseudo-CQT version" }
        require(data.getInt(8) == HEADER_BYTES) { "Invalid LV-Chordia pseudo-CQT header" }

        val sampleRate = data.getDouble(16)
        val hopLength = data.getInt(24)
        val fftSize = data.getInt(28)
        val binCount = data.getInt(32)
        val coefficientCount = data.getInt(36)
        val payloadSize = data.getInt(40)
        val generatorLength = data.getInt(44)
        val generatorOffset = data.getInt(48)
        val rowOffsetsOffset = data.getInt(52)
        val fftBinsOffset = data.getInt(56)
        val coefficientsOffset = data.getInt(60)

        require(HEADER_BYTES + payloadSize == artifact.size) {
            "Invalid LV-Chordia pseudo-CQT payload size"
        }
        val expectedSha = artifact.copyOfRange(64, 96)
        val actualSha = MessageDigest.getInstance("SHA-256")
            .digest(artifact.copyOfRange(HEADER_BYTES, artifact.size))
        require(expectedSha.contentEquals(actualSha)) {
            "LV-Chordia pseudo-CQT payload SHA-256 mismatch"
        }

        require(generatorOffset >= HEADER_BYTES && generatorLength >= 0)
        require(generatorOffset + generatorLength <= artifact.size)
        val generator = artifact.copyOfRange(generatorOffset, generatorOffset + generatorLength)
        require(generator.all { (it.toInt() and 0xff) <= 0x7f }) {
            "LV-Chordia pseudo-CQT generator metadata must be ASCII"
        }

        require(sampleRate == LvChordiaContract.SAMPLE_RATE.toDouble())
        require(hopLength == LvChordiaContract.HOP_LENGTH)
        require(fftSize > 0 && fftSize and (fftSize - 1) == 0) {
            "LV-Chordia pseudo-CQT FFT size must be a power of two"
        }
        require(binCount == LvChordiaContract.PSEUDO_MODEL_BINS)
        require(coefficientCount >= 0)

        val rowOffsets = IntArray(binCount + 1) { index ->
            data.getInt(rowOffsetsOffset + index * Int.SIZE_BYTES)
        }
        val fftBins = IntArray(coefficientCount) { index ->
            data.getInt(fftBinsOffset + index * Int.SIZE_BYTES)
        }
        val coefficients = FloatArray(coefficientCount) { index ->
            data.getFloat(coefficientsOffset + index * Float.SIZE_BYTES)
        }
        require(rowOffsets.first() == 0 && rowOffsets.last() == coefficientCount)
        require(fftBins.all { it in 0..fftSize / 2 })

        return LvChordiaPseudoPlan(
            sampleRate = sampleRate,
            hopLength = hopLength,
            fftSize = fftSize,
            binCount = binCount,
            rowOffsets = rowOffsets,
            fftBins = fftBins,
            coefficients = coefficients,
        )
    }
}

internal class LvChordiaPseudoCqtFrontend(
    private val plan: LvChordiaPseudoPlan,
) {
    data class Features(
        val values: FloatArray,
        val frameCount: Int,
        val binCount: Int,
    )

    private val window = DoubleArray(plan.fftSize) { index ->
        0.5 - 0.5 * cos(2.0 * PI * index / plan.fftSize.toDouble())
    }

    fun transform(audio: FloatArray): Features {
        if (audio.isEmpty()) return Features(FloatArray(0), 0, plan.binCount)
        val frames = 1 + audio.size / plan.hopLength
        val output = FloatArray(frames * plan.binCount)
        val real = DoubleArray(plan.fftSize)
        val imag = DoubleArray(plan.fftSize)
        val spectrum = DoubleArray(plan.fftSize / 2 + 1)
        val half = plan.fftSize / 2

        for (frame in 0 until frames) {
            java.util.Arrays.fill(real, 0.0)
            java.util.Arrays.fill(imag, 0.0)
            val center = frame * plan.hopLength
            val start = center - half
            for (index in real.indices) {
                val source = start + index
                if (source in audio.indices) {
                    real[index] = audio[source].toDouble() * window[index]
                }
            }

            FFT.transform(real, imag)
            for (index in spectrum.indices) {
                spectrum[index] = sqrt(real[index] * real[index] + imag[index] * imag[index])
            }

            for (bin in 0 until plan.binCount) {
                var value = 0.0
                val first = plan.rowOffsets[bin]
                val end = plan.rowOffsets[bin + 1]
                for (coefficient in first until end) {
                    value += plan.coefficients[coefficient] * spectrum[plan.fftBins[coefficient]]
                }
                output[frame * plan.binCount + bin] = value.toFloat()
            }
        }

        return Features(output, frames, plan.binCount)
    }
}
