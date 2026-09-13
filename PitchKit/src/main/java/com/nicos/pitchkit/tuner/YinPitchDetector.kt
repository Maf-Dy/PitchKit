package com.nicos.pitchkit.tuner
// Modified in Maf-Dy/PitchKit fork: DSP correctness, performance, and lifecycle fixes.

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

internal class YinPitchDetector(
    private val sampleRate: Int,
    private val minFrequency: Double,
    private val maxFrequency: Double,
    private val threshold: Double = 0.15,
) {
    private var yin = DoubleArray(0)

    fun detect(buffer: FloatArray): Float {
        if (buffer.size < 32 || minFrequency <= 0.0 || maxFrequency <= minFrequency) return -1f
        val maxTau = min(buffer.size / 2 - 1, floor(sampleRate / minFrequency).toInt())
        val minTau = max(2, floor(sampleRate / maxFrequency).toInt())
        if (maxTau <= minTau) return -1f
        if (yin.size < maxTau + 1) yin = DoubleArray(maxTau + 1)

        val compareLength = min(buffer.size / 2, buffer.size - maxTau)
        if (compareLength <= 0) return -1f

        yin[0] = 1.0
        var t = 1
        while (t <= maxTau) {
            var sum = 0.0
            var i = 0
            while (i < compareLength) {
                val delta = buffer[i] - buffer[i + t]
                sum += delta * delta
                i++
            }
            yin[t] = sum
            t++
        }

        var running = 0.0
        t = 1
        while (t <= maxTau) {
            running += yin[t]
            yin[t] = if (running > 0.0) yin[t] * t / running else 1.0
            t++
        }

        t = minTau
        while (t <= maxTau) {
            if (yin[t] < threshold) {
                while (t + 1 <= maxTau && yin[t + 1] < yin[t]) t++
                val betterTau = parabolicInterp(yin, t, maxTau)
                return (sampleRate / betterTau).toFloat()
            }
            t++
        }
        return -1f
    }

    private fun parabolicInterp(values: DoubleArray, tau: Int, maxTau: Int): Double {
        val x0 = if (tau > 1) tau - 1 else tau
        val x2 = if (tau < maxTau) tau + 1 else tau
        if (x0 == tau || x2 == tau) return tau.toDouble()
        val s0 = values[x0]
        val s1 = values[tau]
        val s2 = values[x2]
        val denom = 2 * (2 * s1 - s2 - s0)
        return if (denom == 0.0) tau.toDouble() else tau + (s2 - s0) / denom
    }
}
