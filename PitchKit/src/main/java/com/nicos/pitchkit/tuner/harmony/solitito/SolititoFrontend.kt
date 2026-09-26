package com.nicos.pitchkit.tuner.harmony.solitito

import com.nicos.pitchkit.tuner.FFT
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * The 168 features solitito's ONNX graph actually takes, one frame at a time.
 *
 * Transcribed from `code/src/audio.rs::compute_cqt_chroma` by way of
 * `tools/accuracy_audit/solitito_live.py`, which is the spec `SolititoParityTest` diffs
 * against. Per 8192-sample frame, at a 256-sample hop:
 *
 *  1. RMS of the **raw** frame — before [SolititoContract.INPUT_GAIN], because `audio.rs`
 *     is explicit that the meter is true input dBFS rather than dBFS plus 6. Below the
 *     gate the frame is 168 zeros and is marked not live.
 *  2. Above it: `x2.0` input gain, a **symmetric** Hann window (`/(N-1)`, not librosa's
 *     periodic `/N` — one of the places a naive reimplementation silently diverges), and
 *     a real FFT to 4097 bins.
 *  3. The sparse pseudo-CQT `|spectrum · K|` over the CSR kernel in [SolititoDspPlan].
 *  4. Bass boost `x5.0` on bins 0..35, then a **per-frame** 80 dB log normalisation
 *     against `max(frameMax, 0.005)`.
 *  5. Chroma through the shipped `cq_to_chroma` matrix, max-normalised; bass energy as
 *     the pairwise mean of the lowest 24 bins; `concat(cqt144, chroma12, bass12)`.
 *
 * Two train/serve gaps are reproduced rather than fixed, because the serving path is what
 * the app ships and what the Stage C screening measured: the trainer normalises globally
 * (`amplitude_to_db(ref=np.max)` over a whole file) where this normalises per frame, and
 * the trainer uses a true multi-resolution `librosa.cqt` where this is one fixed-length
 * FFT times a kernel. Both are upstream's, not this port's.
 *
 * Not thread-safe: the scratch buffers are reused frame to frame. The recognizer that
 * owns it is `@Synchronized`.
 */
internal class SolititoFrontend(
    private val plan: SolititoDspPlan,
    gateDb: Double = SolititoContract.GATE_DB,
    private val bassBoost: Boolean = SolititoContract.BASS_BOOST_ENABLED,
    private val bassBoostGain: Double = SolititoContract.BASS_BOOST_GAIN,
) {
    /** The gate this front end was built with, recorded for the replay provenance. */
    val gateDecibels: Double = gateDb
    private val gate: Float = SolititoContract.dbToLinear(gateDb).toFloat()

    private val size = SolititoContract.FFT_SIZE

    /**
     * Symmetric Hann, `0.5 (1 - cos(2 pi i / (N - 1)))`, held as `Float` because
     * `audio.rs` windows in `f32` and the reference's `np.hanning(...).astype(float32)`
     * does the same.
     */
    private val window = FloatArray(size) { index ->
        (0.5 * (1.0 - cos(2.0 * Math.PI * index / (size - 1)))).toFloat()
    }

    private val real = DoubleArray(size)
    private val imaginary = DoubleArray(size)
    private val magnitude = DoubleArray(plan.binCount)

    /** True when the raw frame RMS clears the gate. */
    fun isLive(rms: Float): Boolean = rms > gate

    /** RMS of one raw 8192-sample frame, accumulated in `Double`. */
    fun frameRms(samples: FloatArray, offset: Int): Float {
        var total = 0.0
        for (index in 0 until size) {
            val value = samples[offset + index].toDouble()
            total += value * value
        }
        return sqrt(total / size).toFloat()
    }

    /**
     * 168 features for the frame starting at [offset]. The caller has already decided the
     * frame is live; a gated frame is 168 zeros and never reaches here.
     */
    fun features(samples: FloatArray, offset: Int, destination: FloatArray) {
        require(destination.size == SolititoContract.FEATURE_COUNT) {
            "destination must hold ${SolititoContract.FEATURE_COUNT} features"
        }
        for (index in 0 until size) {
            // `(chunk * INPUT_GAIN).astype(float32) * window`, in f32 as upstream.
            real[index] = ((samples[offset + index] * SolititoContract.INPUT_GAIN) *
                window[index]).toDouble()
            imaginary[index] = 0.0
        }
        FFT.transform(real, imaginary)

        sparseCqtMagnitude()

        // Upstream holds the CQT magnitudes in f32 from here on, and the reference
        // reproduces that (`np.abs(...).astype(np.float32)`), so round before the boost.
        var peak = 0.0
        val boost = bassBoostGain.toFloat()
        for (bin in 0 until plan.binCount) {
            var value = magnitude[bin].toFloat()
            if (bassBoost && bin < SolititoContract.BASS_BOOST_CUTOFF) value *= boost
            magnitude[bin] = value.toDouble()
            if (magnitude[bin] > peak) peak = magnitude[bin]
        }
        val reference = maxOf(peak, SolititoContract.MIN_REFERENCE_LEVEL)

        // Per-frame 80 dB log normalisation, `20 log10(max(cqt, 1e-9) / reference)`.
        for (bin in 0 until plan.binCount) {
            val decibels = 20.0 * log10(maxOf(magnitude[bin], 1e-9) / reference)
            val normalised = (decibels + SolititoContract.NORMALISATION_DB) /
                SolititoContract.NORMALISATION_DB
            destination[bin] = normalised.coerceIn(0.0, 1.0).toFloat()
        }

        // Chroma: the shipped cq_to_chroma matrix, then max-normalised (norm=inf).
        val chromaBase = plan.binCount
        var chromaPeak = 0.0
        for (chromaBin in 0 until plan.chromaBinCount) {
            var total = 0.0
            for (bin in 0 until plan.binCount) {
                val weight = plan.chroma[bin * plan.chromaBinCount + chromaBin]
                if (weight != 0f) total += destination[bin].toDouble() * weight
            }
            destination[chromaBase + chromaBin] = total.toFloat()
            if (total > chromaPeak) chromaPeak = total
        }
        if (chromaPeak > 1e-9) {
            for (chromaBin in 0 until plan.chromaBinCount) {
                destination[chromaBase + chromaBin] =
                    (destination[chromaBase + chromaBin] / chromaPeak).toFloat()
            }
        }

        // Bass energy: the pairwise mean of the lowest 24 CQT bins, i.e. 12 semitones.
        val bassBase = chromaBase + plan.chromaBinCount
        for (index in 0 until SolititoContract.BASS_BINS) {
            val low = destination[2 * index].toDouble()
            val high = destination[2 * index + 1].toDouble()
            destination[bassBase + index] = (0.5 * (low + high)).toFloat()
        }
    }

    /**
     * `audio.rs::sparse_cqt_mag`, bin by bin over the CSR kernel.
     *
     * 40 675 complex multiply-accumulates per frame at 62.5 frames a second. The
     * accumulation is `Double` where the Rust is `f32`; the screening measured that
     * difference at 1.3e-7 relative, three orders below the 1e-4 prune the kernel already
     * carries.
     */
    private fun sparseCqtMagnitude() {
        val offsets = plan.rowOffsets
        val bins = plan.fftBins
        val weightsReal = plan.weightsReal
        val weightsImaginary = plan.weightsImaginary
        for (bin in 0 until plan.binCount) {
            var sumReal = 0.0
            var sumImaginary = 0.0
            for (index in offsets[bin] until offsets[bin + 1]) {
                val fftBin = bins[index]
                val spectrumReal = real[fftBin]
                // The FFT is in place over a real input, so the rfft bin's imaginary
                // part is the transform's own; bins 0 and N/2 carry zero there.
                val spectrumImaginary = imaginary[fftBin]
                val weightReal = weightsReal[index].toDouble()
                val weightImaginary = weightsImaginary[index].toDouble()
                sumReal += spectrumReal * weightReal - spectrumImaginary * weightImaginary
                sumImaginary += spectrumReal * weightImaginary + spectrumImaginary * weightReal
            }
            // `hypot`, not `sqrt(r² + i²)`: it is what `np.abs` on a complex array does
            // and it does not overflow or lose a bit on the square.
            magnitude[bin] = hypot(sumReal, sumImaginary)
        }
    }
}
