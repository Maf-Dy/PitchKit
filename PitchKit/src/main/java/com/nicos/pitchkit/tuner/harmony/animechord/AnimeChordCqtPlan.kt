package com.nicos.pitchkit.tuner.harmony.animechord

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * The precomputed `RecursiveCQT` front end, as baked by
 * `tools/generate-animechord-cqt-plans.py`.
 *
 * Everything the reference designs at construction time is a constant here, because the
 * reference's own parameters make it one. Stage `i` runs at `22050 / 2^i` Hz with hop
 * `512 / 2^i`, and the ratio `Q * fs / f` that decides both the kernel length and the
 * FFT size is scale invariant, so **every** stage uses a 256-point FFT, the same
 * 256-point periodic hann window and the same 65-tap decimator. Only the seven 36 x 129
 * complex kernels are stored per stage, and even they agree to the last float — they are
 * kept separate anyway so the plan is a transcript of the reference's buffers rather than
 * an argument about them.
 *
 * @property kernels stage-major `[stage][bin][fftBin]` complex pairs, real then
 * imaginary. Stage `i` covers octave `OCTAVES - 1 - i`, i.e. stage 0 is the top octave at
 * the full rate.
 */
internal class AnimeChordCqtPlan(
    val formatVersion: Int,
    val sampleRate: Double,
    val hopLength: Int,
    val binCount: Int,
    val binsPerOctave: Int,
    val octaveCount: Int,
    val stageFftSize: Int,
    val stageBinCount: Int,
    val fmin: Double,
    val q: Double,
    val filterScale: Double,
    val cropNFft: Int,
    /** 256 float analysis window, periodic hann. */
    val window: FloatArray,
    /** `octaveCount * binsPerOctave * stageBinCount * 2` floats. */
    val kernels: FloatArray,
    /** 65 taps, `firwin(65, sr/4, fs=sr, window=("kaiser", 5.0))`. */
    val decimator: FloatArray,
    val payloadSha256: String,
)

internal object AnimeChordCqtPlanDecoder {
    private const val MAGIC = 0x54514341
    private const val HEADER_BYTES = 128

    fun decodeAndVerify(artifact: ByteArray): AnimeChordCqtPlan {
        require(artifact.size >= HEADER_BYTES) { "anime CQT plan is shorter than its header" }
        val data = ByteBuffer.wrap(artifact).order(ByteOrder.LITTLE_ENDIAN)
        require(data.getInt(0) == MAGIC) { "Invalid anime CQT plan magic" }
        require(data.getInt(8) == HEADER_BYTES) { "Unsupported anime CQT plan header size" }

        val formatVersion = data.getInt(4)
        require(formatVersion == 1) { "Unsupported anime CQT plan version $formatVersion" }

        val octaveCount = data.getInt(12)
        val sampleRate = data.getDouble(16)
        val hopLength = data.getInt(24)
        val binCount = data.getInt(28)
        val binsPerOctave = data.getInt(32)
        val stageFftSize = data.getInt(36)
        val fmin = data.getDouble(40)
        val q = data.getDouble(48)
        val filterScale = data.getDouble(56)
        val decimatorTaps = data.getInt(64)
        val payloadSize = data.getInt(68)
        val windowOffset = data.getInt(72)
        val kernelOffset = data.getInt(76)
        val decimatorOffset = data.getInt(80)
        val cropNFft = data.getInt(84)

        require(HEADER_BYTES + payloadSize == artifact.size) {
            "Invalid anime CQT plan payload size"
        }
        val expected = artifact.copyOfRange(96, 128)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        val actual = sha256(artifact.copyOfRange(HEADER_BYTES, artifact.size))
        require(actual == expected) { "anime CQT plan payload SHA-256 mismatch" }

        require(octaveCount > 0 && binsPerOctave > 0)
        require(binCount == octaveCount * binsPerOctave) {
            "anime CQT plan bins $binCount != $octaveCount x $binsPerOctave"
        }
        require(stageFftSize > 0 && stageFftSize and (stageFftSize - 1) == 0) {
            "anime CQT stage FFT size must be a power of two"
        }
        require(decimatorTaps > 0 && decimatorTaps % 2 == 1) {
            "the decimator must have an odd tap count so its delay is an integer"
        }
        val stageBinCount = stageFftSize / 2 + 1

        return AnimeChordCqtPlan(
            formatVersion = formatVersion,
            sampleRate = sampleRate,
            hopLength = hopLength,
            binCount = binCount,
            binsPerOctave = binsPerOctave,
            octaveCount = octaveCount,
            stageFftSize = stageFftSize,
            stageBinCount = stageBinCount,
            fmin = fmin,
            q = q,
            filterScale = filterScale,
            cropNFft = cropNFft,
            window = readFloats(data, windowOffset, stageFftSize),
            kernels = readFloats(
                data,
                kernelOffset,
                octaveCount * binsPerOctave * stageBinCount * 2,
            ),
            decimator = readFloats(data, decimatorOffset, decimatorTaps),
            payloadSha256 = expected,
        )
    }

    private fun readFloats(buffer: ByteBuffer, offset: Int, count: Int): FloatArray =
        FloatArray(count) { index -> buffer.getFloat(offset + index * Float.SIZE_BYTES) }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
