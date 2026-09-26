package com.nicos.pitchkit.tuner.harmony.animechord

import com.nicos.pitchkit.tuner.FFT
import kotlin.math.ceil
import kotlin.math.sqrt

internal class AnimeChordFeatures(
    /** Row-major `[frameCount x INPUT_BINS]`, standardised over the whole song. */
    val values: FloatArray,
    val frameCount: Int,
    /** `(samples - 2048) / 512 + 1`, the frames the label grid actually covers. */
    val cropLength: Int,
)

/**
 * Android port of `chord_transcription/models/cqt.py::RecursiveCQT`, plus the whole-clip
 * standardisation `AudioFeatureExtractor` applies on top of it.
 *
 * It is **not** a mel spectrogram and **not** librosa's CQT. Seven stages run the same
 * 256-point STFT at halving rates — stage `i` at `22050 / 2^i` Hz with hop `512 / 2^i`,
 * `center=True` and reflect padding — and each multiplies its 129 rfft bins by a fixed
 * 36 x 129 complex kernel of L1-normalised hann-windowed complex exponentials. Between
 * stages the signal is halved by a 65-tap Kaiser(beta=5) low-pass at `sr/4`, applied as a
 * stride-2 correlation with 32 samples of zero padding, exactly as the reference's
 * `nn.Conv1d` does. Octaves are stacked low to high, each padded or trimmed to
 * `ceil(samples / 512)` frames, and the magnitude is taken at the end.
 *
 * Standardisation is `(x - mean) / (std + 1e-8)` per channel over the whole (freq, time)
 * block, with torch's unbiased standard deviation. The statistics are **whole-song**:
 * `.accuracy-work/annotations/anime-windowed-report.md` §7.4 measured per-window
 * statistics at 1.5 family points and 2.5 exact points worse, and the whole-song pass is
 * free because the offline analyzer already holds the matrix.
 *
 * The app feeds mono, so the stereo channels the graph wants are one channel duplicated
 * and only one is ever computed. The PC reference lane read the files as stereo; that is
 * the same mono/stereo difference every other engine in this app already carries.
 */
internal class AnimeChordCqtFrontend(
    planBytes: ByteArray,
) {
    private val plan = AnimeChordCqtPlanDecoder.decodeAndVerify(planBytes)

    init {
        require(plan.sampleRate == AnimeChordContract.SAMPLE_RATE.toDouble())
        require(plan.hopLength == AnimeChordContract.HOP_LENGTH)
        require(plan.binCount == AnimeChordContract.INPUT_BINS)
        require(plan.binsPerOctave == AnimeChordContract.BINS_PER_OCTAVE)
        require(plan.octaveCount == AnimeChordContract.OCTAVES)
        require(plan.stageFftSize == AnimeChordContract.STAGE_FFT)
        require(plan.stageBinCount == AnimeChordContract.STAGE_BINS)
        require(plan.cropNFft == AnimeChordContract.CROP_N_FFT)
    }

    fun transform(audio: FloatArray): AnimeChordFeatures {
        val samples = audio.size
        if (samples <= 0) return AnimeChordFeatures(FloatArray(0), 0, 0)

        val bins = plan.binCount
        val frameCount = ceil(samples.toDouble() / plan.hopLength).toInt()
        if (frameCount <= 0) return AnimeChordFeatures(FloatArray(0), 0, 0)
        val output = FloatArray(frameCount * bins)

        var signal = audio
        for (stage in 0 until plan.octaveCount) {
            transformStage(signal, stage, frameCount, output)
            if (stage < plan.octaveCount - 1) signal = decimate(signal)
        }

        val cropLength = (samples - plan.cropNFft) / plan.hopLength + 1
        standardise(output)
        return AnimeChordFeatures(
            values = output,
            frameCount = frameCount,
            cropLength = cropLength.coerceIn(0, frameCount),
        )
    }

    /**
     * One STFT stage. The reference pads each octave's complex output out to
     * `target_frames` with zeros before taking the magnitude, so a stage that produces
     * fewer frames than the grid simply leaves the tail at zero.
     */
    private fun transformStage(
        input: FloatArray,
        stage: Int,
        targetFrames: Int,
        output: FloatArray,
    ) {
        if (input.isEmpty()) return
        val fftSize = plan.stageFftSize
        val pad = fftSize / 2
        val hop = plan.hopLength shr stage
        check(hop >= 1) { "hop collapsed to zero at stage $stage" }
        val stageFrames = minOf(targetFrames, 1 + input.size / hop)
        val binsPerOctave = plan.binsPerOctave
        // Octaves are emitted low to high, and stage 0 is the *top* octave.
        val binOffset = (plan.octaveCount - 1 - stage) * binsPerOctave
        val kernelBase = stage * binsPerOctave * plan.stageBinCount * 2

        val real = DoubleArray(fftSize)
        val imaginary = DoubleArray(fftSize)
        for (frame in 0 until stageFrames) {
            val start = frame * hop - pad
            for (offset in 0 until fftSize) {
                real[offset] = reflect(input, start + offset) * plan.window[offset].toDouble()
                imaginary[offset] = 0.0
            }
            FFT.transform(real, imaginary)

            val destination = frame * plan.binCount + binOffset
            for (bin in 0 until binsPerOctave) {
                var row = kernelBase + bin * plan.stageBinCount * 2
                var sumReal = 0.0
                var sumImaginary = 0.0
                for (fftBin in 0 until plan.stageBinCount) {
                    val coefficientReal = plan.kernels[row].toDouble()
                    val coefficientImaginary = plan.kernels[row + 1].toDouble()
                    row += 2
                    val sampleReal = real[fftBin]
                    val sampleImaginary = imaginary[fftBin]
                    sumReal += coefficientReal * sampleReal - coefficientImaginary * sampleImaginary
                    sumImaginary += coefficientReal * sampleImaginary +
                        coefficientImaginary * sampleReal
                }
                output[destination + bin] =
                    sqrt(sumReal * sumReal + sumImaginary * sumImaginary).toFloat()
            }
        }
    }

    /**
     * `pad_mode="reflect"`: the edge sample is a mirror axis and is not repeated. A clip
     * shorter than the 128-sample pad cannot be reflected, so the index is clamped —
     * torch would refuse outright, and the analyzer never feeds a window that short.
     */
    private fun reflect(input: FloatArray, index: Int): Double {
        val last = input.size - 1
        if (last <= 0) return input[0].toDouble()
        var position = index
        if (position < 0) position = -position
        if (position > last) position = 2 * last - position
        if (position < 0 || position > last) position = position.coerceIn(0, last)
        return input[position].toDouble()
    }

    /**
     * `nn.Conv1d(1, 1, 65, stride = 2, padding = 32, bias = False)` with the firwin taps
     * as weights. PyTorch convolution is a cross-correlation; the taps are symmetric, so
     * the distinction does not matter, but it is written the way the graph runs it.
     */
    private fun decimate(input: FloatArray): FloatArray {
        val taps = plan.decimator
        val delay = (taps.size - 1) / 2
        val count = (input.size + 1) / 2
        val output = FloatArray(count)
        for (index in 0 until count) {
            val base = index * 2 - delay
            var value = 0.0
            var source = base
            for (tap in taps.indices) {
                if (source in input.indices) value += input[source] * taps[tap].toDouble()
                source++
            }
            output[index] = value.toFloat()
        }
        return output
    }

    /** `(x - mean) / (std + 1e-8)`, torch's unbiased std, over the whole block. */
    private fun standardise(values: FloatArray) {
        if (values.isEmpty()) return
        var sum = 0.0
        for (value in values) sum += value.toDouble()
        val mean = sum / values.size
        var squared = 0.0
        for (value in values) {
            val delta = value.toDouble() - mean
            squared += delta * delta
        }
        val divisor = (values.size - 1).coerceAtLeast(1)
        val deviation = sqrt(squared / divisor) + AnimeChordContract.STANDARDISATION_EPSILON
        for (index in values.indices) {
            values[index] = ((values[index].toDouble() - mean) / deviation).toFloat()
        }
    }
}
