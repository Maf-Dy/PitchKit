package com.nicos.pitchkit.tuner.harmony.chordnet

import com.nicos.pitchkit.tuner.FFT
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.sqrt

/** CPU implementation of the plan-driven recursive librosa CQT. */
class CqtCpuFrontend(
    private val plan: CqtPlan,
) {
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

        for (octave in plan.octaves.sortedBy { it.index }) {
            val targetDownsampleCount = plan.earlyDownsampleCount + octave.index
            while (downsampleCount < targetDownsampleCount) {
                current = downsampleByTwo(current)
                downsampleCount++
            }
            processOctave(current, octave, frameCount, output)
        }

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

    private fun downsampleByTwo(input: FloatArray): FloatArray {
        val outputCount = ceil(input.size / 2.0).toInt()
        val output = FloatArray(outputCount)
        val downsample = plan.downsample
        val delay = downsample.delay

        for (outputIndex in output.indices) {
            val center = outputIndex * 2
            var value = 0.0
            for (tap in 0 until downsample.tapCount) {
                val sourceIndex = center + tap - delay
                if (sourceIndex !in input.indices) continue
                val distance = kotlin.math.abs(tap - delay)
                value += input[sourceIndex] * downsample.halfCoefficients[distance]
            }
            output[outputIndex] = (value * downsample.gain).toFloat()
        }
        return output
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
