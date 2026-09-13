package com.nicos.pitchkit.tuner

import kotlin.math.ceil
import kotlin.math.floor

internal class YinPitchDetector(
    private val sampleRate: Int,
    private val minFrequencyHz: Double,
    private val maxFrequencyHz: Double,
    private val threshold: Double = 0.15,
) {
    private var yinScratch = DoubleArray(0)

    init {
        require(sampleRate > 0) { "sampleRate must be > 0" }
        require(minFrequencyHz > 0.0) { "minFrequencyHz must be > 0" }
        require(maxFrequencyHz > minFrequencyHz) {
            "maxFrequencyHz must be greater than minFrequencyHz"
        }
    }

    fun detect(buffer: FloatArray): Float {
        if (buffer.size < 4) return -1f

        val minTau = floor(sampleRate / maxFrequencyHz)
            .toInt()
            .coerceAtLeast(2)
        val maxTau = ceil(sampleRate / minFrequencyHz)
            .toInt()
            .coerceAtMost(buffer.size - 2)
        if (minTau > maxTau) return -1f

        ensureScratch(maxTau + 2)
        java.util.Arrays.fill(yinScratch, 0, maxTau + 2, 0.0)

        val comparisonLength = buffer.size - maxTau
        if (comparisonLength < 2) return -1f

        for (tau in 1..maxTau) {
            var sum = 0.0
            var i = 0
            while (i < comparisonLength) {
                val delta = buffer[i] - buffer[i + tau]
                sum += delta * delta
                i++
            }
            yinScratch[tau] = sum
        }

        yinScratch[0] = 1.0
        var runningSum = 0.0
        for (tau in 1..maxTau) {
            runningSum += yinScratch[tau]
            if (runningSum != 0.0) {
                yinScratch[tau] *= tau / runningSum
            }
        }

        var estimate = -1
        var tau = minTau
        while (tau <= maxTau) {
            if (yinScratch[tau] < threshold) {
                while (tau + 1 <= maxTau && yinScratch[tau + 1] < yinScratch[tau]) {
                    tau++
                }
                estimate = tau
                break
            }
            tau++
        }
        if (estimate == -1) return -1f

        val betterTau = parabolicInterpolation(yinScratch, estimate, maxTau)
        return if (betterTau > 0.0) (sampleRate / betterTau).toFloat() else -1f
    }

    private fun ensureScratch(requiredSize: Int) {
        if (yinScratch.size < requiredSize) {
            yinScratch = DoubleArray(requiredSize)
        }
    }

    private fun parabolicInterpolation(
        yin: DoubleArray,
        tau: Int,
        maxTau: Int,
    ): Double {
        val x0 = if (tau > 0) tau - 1 else tau
        val x2 = if (tau + 1 <= maxTau) tau + 1 else tau
        if (x0 == tau) return if (yin[tau] <= yin[x2]) tau.toDouble() else x2.toDouble()
        if (x2 == tau) return if (yin[tau] <= yin[x0]) tau.toDouble() else x0.toDouble()
        val s0 = yin[x0]
        val s1 = yin[tau]
        val s2 = yin[x2]
        val denominator = 2 * (2 * s1 - s2 - s0)
        return if (denominator == 0.0) tau.toDouble() else tau + (s2 - s0) / denominator
    }
}
