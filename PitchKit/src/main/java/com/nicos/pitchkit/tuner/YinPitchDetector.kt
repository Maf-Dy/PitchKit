package com.nicos.pitchkit.tuner

internal class YinPitchDetector(
    private val sampleRate: Int,
    private val threshold: Double = 0.15,
) {
    fun detect(buffer: FloatArray): Float {
        if (buffer.size < 4) return -1f
        val tau = buffer.size / 2
        val yin = DoubleArray(tau)

        for (t in 1 until tau) {
            var sum = 0.0
            for (i in 0 until tau) {
                val delta = buffer[i] - buffer[i + t]
                sum += delta * delta
            }
            yin[t] = sum
        }

        yin[0] = 1.0
        var runningSum = 0.0
        for (t in 1 until tau) {
            runningSum += yin[t]
            if (runningSum != 0.0) yin[t] *= t / runningSum
        }

        var estimate = -1
        var t = 2
        while (t < tau) {
            if (yin[t] < threshold) {
                while (t + 1 < tau && yin[t + 1] < yin[t]) t++
                estimate = t
                break
            }
            t++
        }
        if (estimate == -1) return -1f

        val betterTau = parabolicInterpolation(yin, estimate)
        return if (betterTau > 0.0) (sampleRate / betterTau).toFloat() else -1f
    }

    private fun parabolicInterpolation(yin: DoubleArray, tau: Int): Double {
        val x0 = if (tau > 0) tau - 1 else tau
        val x2 = if (tau + 1 < yin.size) tau + 1 else tau
        if (x0 == tau) return if (yin[tau] <= yin[x2]) tau.toDouble() else x2.toDouble()
        if (x2 == tau) return if (yin[tau] <= yin[x0]) tau.toDouble() else x0.toDouble()
        val s0 = yin[x0]
        val s1 = yin[tau]
        val s2 = yin[x2]
        val denominator = 2 * (2 * s1 - s2 - s0)
        return if (denominator == 0.0) tau.toDouble() else tau + (s2 - s0) / denominator
    }
}
