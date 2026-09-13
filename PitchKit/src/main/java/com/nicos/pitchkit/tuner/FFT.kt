package com.nicos.pitchkit.tuner

import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

internal object FFT {
    /** Reusable storage for callers doing the same transform shape repeatedly. */
    class MagnitudeWorkspace {
        private var real = DoubleArray(0)
        private var imaginary = DoubleArray(0)
        private var magnitude = DoubleArray(0)

        fun magnitudePadded(samples: FloatArray, padFactor: Int = 2): DoubleArray {
            if (samples.size < 2) return DoubleArray(0)
            require(padFactor >= 1) { "padFactor must be >= 1" }

            val base = Integer.highestOneBit(samples.size)
            val n = base * padFactor
            require(n and (n - 1) == 0) {
                "base * padFactor must be a power of two"
            }
            ensureSize(n)
            java.util.Arrays.fill(real, 0.0)
            java.util.Arrays.fill(imaginary, 0.0)

            for (i in 0 until base) {
                val window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (base - 1)))
                real[i] = samples[i] * window
            }

            transform(real, imaginary)
            for (index in magnitude.indices) {
                magnitude[index] = hypot(real[index], imaginary[index])
            }
            return magnitude
        }

        private fun ensureSize(n: Int) {
            if (real.size == n) return
            real = DoubleArray(n)
            imaginary = DoubleArray(n)
            magnitude = DoubleArray(n / 2)
        }
    }

    // Pitch analysis already runs on worker threads. Reuse the large FFT arrays on
    // each worker without sharing a mutable result buffer across analyzer threads.
    private val magnitudeWorkspace = ThreadLocal.withInitial { MagnitudeWorkspace() }

    fun transform(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        require(im.size == n) { "real and imaginary arrays must have equal length" }
        if (n == 1) return
        require(n and (n - 1) == 0) { "length must be power of 2" }

        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                re[i] = re[j].also { re[j] = re[i] }
                im[i] = im[j].also { im[j] = im[i] }
            }
        }

        var length = 2
        while (length <= n) {
            val angle = -2.0 * Math.PI / length
            val wRe = cos(angle)
            val wIm = sin(angle)
            var i = 0
            while (i < n) {
                var currentRe = 1.0
                var currentIm = 0.0
                for (k in 0 until length / 2) {
                    val aRe = re[i + k]
                    val aIm = im[i + k]
                    val bRe = re[i + k + length / 2] * currentRe -
                        im[i + k + length / 2] * currentIm
                    val bIm = re[i + k + length / 2] * currentIm +
                        im[i + k + length / 2] * currentRe
                    re[i + k] = aRe + bRe
                    im[i + k] = aIm + bIm
                    re[i + k + length / 2] = aRe - bRe
                    im[i + k + length / 2] = aIm - bIm
                    val nextRe = currentRe * wRe - currentIm * wIm
                    currentIm = currentRe * wIm + currentIm * wRe
                    currentRe = nextRe
                }
                i += length
            }
            length = length shl 1
        }
    }

    fun magnitudePadded(samples: FloatArray, padFactor: Int = 2): DoubleArray =
        magnitudeWorkspace.get().magnitudePadded(samples, padFactor)

    fun interpolatePeak(magnitudes: DoubleArray, bin: Int): Double {
        if (bin <= 0 || bin >= magnitudes.size - 1) return 0.0
        val a = magnitudes[bin - 1]
        val b = magnitudes[bin]
        val c = magnitudes[bin + 1]
        val denominator = a - 2 * b + c
        return if (denominator == 0.0) 0.0 else 0.5 * (a - c) / denominator
    }
}
