package com.nicos.pitchkit.tuner.harmony.solitito

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * solitito's pseudo-CQT kernel and chroma matrix, decoded from `solitito-dsp.bin`.
 *
 * The kernel is `librosa.filters.constant_q(sr=16000, fmin=C1, n_bins=144,
 * bins_per_octave=24)` moved into the 8192-point FFT domain, conjugated, peak-normalised
 * and pruned at 1e-4 of peak — 40 675 surviving complex weights, 6.9 % dense — stored
 * **CSR by CQT bin**. [rowOffsets] has `nBins + 1` entries and
 * `fftBins[rowOffsets[i] until rowOffsets[i + 1]]` are the FFT bins CQT bin `i` reads.
 *
 * The sparse form is kept rather than expanded: dense would be 4097 x 144 complex64 =
 * 4.7 MB of resident float, against 40 675 taps actually touched per frame.
 */
class SolititoDspPlan(
    val formatVersion: Int,
    val sampleRate: Int,
    val fftSize: Int,
    val fftBinCount: Int,
    val binCount: Int,
    val chromaBinCount: Int,
    /** `binCount + 1` CSR row offsets into [fftBins]/[weightsReal]/[weightsImaginary]. */
    val rowOffsets: IntArray,
    val fftBins: IntArray,
    val weightsReal: FloatArray,
    val weightsImaginary: FloatArray,
    /** Row-major `[binCount][chromaBinCount]`; upstream folds each CQT bin into one class. */
    val chroma: FloatArray,
    val payloadSha256: String,
) {
    val nonZeroCount: Int get() = fftBins.size
}

/** Decodes and verifies the packed plan written by `export_solitito_assets.py`. */
object SolititoDspPlanDecoder {
    /** Little-endian `'S' 'L' 'T' 'D'`. */
    private const val MAGIC = 0x44544C53
    private const val HEADER_BYTES = 128

    fun decodeAndVerify(artifact: ByteArray): SolititoDspPlan {
        require(artifact.size >= HEADER_BYTES) { "solitito DSP plan is shorter than its header" }
        val data = ByteBuffer.wrap(artifact).order(ByteOrder.LITTLE_ENDIAN)
        require(data.getInt(0) == MAGIC) { "Invalid solitito DSP plan magic" }

        val formatVersion = data.getInt(4)
        require(formatVersion == 1) { "Unsupported solitito DSP plan version $formatVersion" }
        require(data.getInt(8) == HEADER_BYTES) { "Unsupported solitito DSP plan header size" }

        val sampleRate = data.getInt(12)
        val fftSize = data.getInt(16)
        val fftBinCount = data.getInt(20)
        val binCount = data.getInt(24)
        val chromaBinCount = data.getInt(28)
        val nonZeroCount = data.getInt(32)
        val payloadBytes = data.getInt(36)
        val rowOffsetsAt = data.getInt(40)
        val fftBinsAt = data.getInt(44)
        val realAt = data.getInt(48)
        val imaginaryAt = data.getInt(52)
        val chromaAt = data.getInt(56)

        require(sampleRate == SolititoContract.SAMPLE_RATE) {
            "solitito DSP plan is for $sampleRate Hz, expected ${SolititoContract.SAMPLE_RATE}"
        }
        require(fftSize == SolititoContract.FFT_SIZE) {
            "solitito DSP plan is for a $fftSize-point FFT, expected ${SolititoContract.FFT_SIZE}"
        }
        require(fftBinCount == SolititoContract.FFT_BINS) {
            "solitito DSP plan has $fftBinCount FFT bins, expected ${SolititoContract.FFT_BINS}"
        }
        require(binCount == SolititoContract.CQT_BINS) {
            "solitito DSP plan has $binCount CQT bins, expected ${SolititoContract.CQT_BINS}"
        }
        require(chromaBinCount == SolititoContract.CHROMA_BINS) {
            "solitito DSP plan has $chromaBinCount chroma bins, expected " +
                "${SolititoContract.CHROMA_BINS}"
        }
        require(nonZeroCount > 0) { "solitito DSP plan carries no kernel weights" }
        require(HEADER_BYTES + payloadBytes == artifact.size) {
            "solitito DSP plan payload is $payloadBytes B in a ${artifact.size} B artifact"
        }

        val expectedPayloadSha = artifact.copyOfRange(64, 96)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        val actualPayloadSha = sha256(artifact.copyOfRange(HEADER_BYTES, artifact.size))
        require(actualPayloadSha == expectedPayloadSha) {
            "solitito DSP plan payload SHA-256 mismatch"
        }

        val rowOffsets = readIntArray(data, rowOffsetsAt, binCount + 1)
        val fftBins = readIntArray(data, fftBinsAt, nonZeroCount)
        val real = readFloatArray(data, realAt, nonZeroCount)
        val imaginary = readFloatArray(data, imaginaryAt, nonZeroCount)
        val chroma = readFloatArray(data, chromaAt, binCount * chromaBinCount)

        require(rowOffsets.first() == 0) { "solitito CSR offsets do not start at 0" }
        require(rowOffsets.last() == nonZeroCount) {
            "solitito CSR offsets end at ${rowOffsets.last()}, expected $nonZeroCount"
        }
        for (index in 0 until binCount) {
            require(rowOffsets[index] <= rowOffsets[index + 1]) {
                "solitito CSR offsets are not non-decreasing at bin $index"
            }
        }
        for (bin in fftBins) {
            require(bin in 0 until fftBinCount) { "A solitito CQT weight points past the FFT" }
        }

        return SolititoDspPlan(
            formatVersion = formatVersion,
            sampleRate = sampleRate,
            fftSize = fftSize,
            fftBinCount = fftBinCount,
            binCount = binCount,
            chromaBinCount = chromaBinCount,
            rowOffsets = rowOffsets,
            fftBins = fftBins,
            weightsReal = real,
            weightsImaginary = imaginary,
            chroma = chroma,
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
