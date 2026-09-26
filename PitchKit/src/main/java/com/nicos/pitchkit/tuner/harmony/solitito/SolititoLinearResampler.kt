package com.nicos.pitchkit.tuner.harmony.solitito

import kotlin.math.floor

/**
 * solitito's own resampler to 16 kHz: bare linear interpolation, **no anti-aliasing**.
 *
 * `audio.rs` walks a read position by `source_rate / 16000` and emits
 * `s0 + frac * (s1 - s0)`; everything above 8 kHz folds straight back into the band. That
 * is a deliberate transcription, not an oversight here: the Stage C part 2 screening
 * priced a properly band-limited `resample_poly` against it over the whole GuitarSet
 * slice and it was worth **+0.3 points** of family accuracy
 * (`live-solitito-report.md` §5.1). The model was trained on `librosa.load`'s filtered
 * resample and, on room-mic guitar, does not care. A polyphase FIR would cost phone CPU
 * for nothing.
 *
 * Streaming without drift. The output index is an absolute counter and every source
 * position is recomputed as `outputIndex * ratio` in `Double`, so the sample grid is a
 * pure function of the output index and is identical however the input is chunked — and
 * identical to the reference's `numpy.arange(count) * ratio`. (The Rust *accumulates* the
 * position in `f32`, whose spacing at 1.3 M samples is 0.125 samples; reproducing that
 * would reproduce a drift the author did not intend.) The arithmetic of the interpolation
 * itself is `Float`, as it is upstream.
 *
 * The emission rule is also the reference's, exactly: sample `n` is emitted once
 * `n * ratio + 1.0 < totalSourceSamples`, i.e. once both taps it reads have arrived under
 * the same bound `resample_linear` uses to size its output.
 */
internal class SolititoLinearResampler(
    private val targetRate: Int = SolititoContract.SAMPLE_RATE,
) {
    private var sourceRate = -1
    private var ratio = 1.0

    /** Source samples not yet consumed, oldest first. */
    private var pending = FloatArray(0)

    /** Absolute index of `pending[0]` in the source stream. */
    private var pendingBase = 0L

    /** Absolute count of source samples seen, and of output samples produced. */
    private var sourceSeen = 0L
    private var outputIndex = 0L

    /** The rate the last [process] call was configured for, or -1. */
    val configuredRate: Int get() = sourceRate

    fun process(input: FloatArray, inputRate: Int): FloatArray {
        require(inputRate > 0) { "inputRate must be > 0" }
        if (inputRate != sourceRate) {
            reset()
            sourceRate = inputRate
            ratio = inputRate.toDouble() / targetRate
        }
        if (input.isEmpty()) return FloatArray(0)

        // A 16 kHz source is a passthrough upstream, so it is one here too.
        if (inputRate == targetRate) {
            sourceSeen += input.size
            outputIndex += input.size
            return input.copyOf()
        }

        append(input)

        val output = ArrayList<Float>(((input.size / ratio).toInt() + 2).coerceAtLeast(1))
        while (true) {
            val position = outputIndex.toDouble() * ratio
            if (position + 1.0 >= sourceSeen.toDouble()) break
            val left = floor(position).toLong()
            val offset = (left - pendingBase).toInt()
            if (offset < 0 || offset + 1 >= pending.size) break
            val fraction = (position - left).toFloat()
            val a = pending[offset]
            val b = pending[offset + 1]
            output.add(a + fraction * (b - a))
            outputIndex++
        }

        dropConsumed()
        return FloatArray(output.size) { output[it] }
    }

    fun reset() {
        sourceRate = -1
        ratio = 1.0
        pending = FloatArray(0)
        pendingBase = 0L
        sourceSeen = 0L
        outputIndex = 0L
    }

    private fun append(input: FloatArray) {
        val combined = FloatArray(pending.size + input.size)
        pending.copyInto(combined)
        input.copyInto(combined, destinationOffset = pending.size)
        pending = combined
        sourceSeen += input.size
    }

    /**
     * Keep only from the next output sample's left tap onward.
     *
     * The drop is clamped to what has actually arrived. When the ratio is above 1 the
     * next sample's left tap can sit *past* the end of the input seen so far — at 44.1 kHz
     * the loop stops on `n * ratio + 1 >= seen`, which leaves `floor(n * ratio)` up to two
     * samples beyond it — and dropping that many would break the invariant
     * `pendingBase + pending.size == sourceSeen` and silently shift every later tap.
     */
    private fun dropConsumed() {
        val nextLeft = floor(outputIndex.toDouble() * ratio).toLong()
        val drop = minOf(nextLeft - pendingBase, pending.size.toLong()).toInt()
        if (drop <= 0) return
        pending = pending.copyOfRange(drop, pending.size)
        pendingBase += drop.toLong()
    }
}
