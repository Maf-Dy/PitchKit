package com.nicos.pitchkit.tuner.harmony.chordnet

import com.nicos.pitchkit.tuner.FFT
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.sqrt

/** CPU implementation of the plan-driven recursive librosa CQT. */
class CqtCpuFrontend(
    private val plan: CqtPlan,
    /** Optional aggregate timings per transform; callback can run on a frontend worker. */
    private val stageTiming: ((String, Long) -> Unit)? = null,
) {
    private val downsampler = CqtDownsampler(plan.downsample)
    private val orderedOctaves = plan.octaves.sortedBy { it.index }
    init {
        require(plan.octaves.isNotEmpty())
        require(plan.config.sampleRate > 0.0)
        require(plan.config.nBins > 0)
    }

    data class Features(
        val values: FloatArray,
        val frameCount: Int,
        val binCount: Int,
    )

    fun transform(audio: FloatArray): Features {
        require(audio.size >= (1 shl plan.earlyDownsampleCount)) {
            "Audio is too short for the CQT plan"
        }

        val frameCount = getCqtFrameCount(audio.size)
        val output = FloatArray(frameCount * plan.config.nBins)

        var current = audio
        var downsampleCount = 0
        var downsampleNanos = 0L
        var projectionNanos = 0L

        for (octave in orderedOctaves) {
            val targetDownsampleCount = plan.earlyDownsampleCount + octave.index
            while (downsampleCount < targetDownsampleCount) {
                val started = if (stageTiming != null) System.nanoTime() else 0L
                current = downsampler.transform(current)
                if (stageTiming != null) downsampleNanos += System.nanoTime() - started
                downsampleCount++
            }
            val started = if (stageTiming != null) System.nanoTime() else 0L
            processOctave(current, octave, frameCount, output)
            if (stageTiming != null) projectionNanos += System.nanoTime() - started
        }
        stageTiming?.invoke("DownsampleMs", downsampleNanos)
        stageTiming?.invoke("ProjectionMs", projectionNanos)

        return Features(output, frameCount, plan.config.nBins)
    }

    private fun processOctave(
        input: FloatArray,
        octave: CqtOctavePlan,
        frameCount: Int,
        output: FloatArray,
    ) {
        val fftSize = octave.fftSize
        val real = DoubleArray(fftSize)
        val imag = DoubleArray(fftSize)
        val half = fftSize / 2

        for (frame in 0 until frameCount) {
            java.util.Arrays.fill(real, 0.0)
            java.util.Arrays.fill(imag, 0.0)

            val center = frame * octave.hopLength
            val start = center - half
            for (sampleInFrame in 0 until fftSize) {
                val sourceIndex = start + sampleInFrame
                if (sourceIndex in input.indices) {
                    real[sampleInFrame] = input[sourceIndex].toDouble()
                }
            }

            FFT.transform(real, imag)

            for (localBin in 0 until octave.binCount) {
                val globalBin = octave.binStart + localBin
                val coefficientStart = plan.rowOffsets[globalBin]
                val coefficientEnd = plan.rowOffsets[globalBin + 1]
                var sumReal = 0.0
                var sumImag = 0.0

                for (coefficientIndex in coefficientStart until coefficientEnd) {
                    val fftBin = plan.fftBins[coefficientIndex]
                    if (fftBin !in real.indices) continue

                    val coefficientReal = plan.coefficients[2 * coefficientIndex].toDouble()
                    val coefficientImag = plan.coefficients[2 * coefficientIndex + 1].toDouble()
                    val sampleReal = real[fftBin]
                    val sampleImag = imag[fftBin]

                    sumReal += coefficientReal * sampleReal - coefficientImag * sampleImag
                    sumImag += coefficientReal * sampleImag + coefficientImag * sampleReal
                }

                val magnitude = sqrt(sumReal * sumReal + sumImag * sumImag)
                output[frame * plan.config.nBins + globalBin] = if (plan.config.logMagnitude) {
                    ln(magnitude + 1e-6).toFloat()
                } else {
                    magnitude.toFloat()
                }
            }
        }
    }

    private fun getCqtFrameCount(sampleCount: Int): Int {
        var minimum = Int.MAX_VALUE
        for (octave in plan.octaves) {
            var octaveSamples = sampleCount
            repeat(plan.earlyDownsampleCount + octave.index) {
                octaveSamples = ceil(octaveSamples / 2.0).toInt()
            }
            val frames = 1 + floor(octaveSamples.toDouble() / octave.hopLength).toInt()
            minimum = minOf(minimum, frames)
        }
        return minimum
    }
}
