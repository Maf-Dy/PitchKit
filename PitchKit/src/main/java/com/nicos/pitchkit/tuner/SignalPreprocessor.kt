package com.nicos.pitchkit.tuner

import kotlin.math.sqrt

/** Shared allocation-free signal cleanup used by realtime and offline analysis. */
internal object SignalPreprocessor {
    fun process(input: FloatArray, output: FloatArray): Double {
        require(input.size == output.size)
        if (input.isEmpty()) return 0.0

        var mean = 0.0
        var i = 0
        while (i < input.size) {
            mean += input[i]
            i++
        }
        mean /= input.size

        var previousInput = 0f
        var previousOutput = 0f
        var energy = 0.0
        val alpha = 0.95f
        i = 0
        while (i < input.size) {
            val centered = input[i] - mean.toFloat()
            val highPassed = alpha * (previousOutput + centered - previousInput)
            previousInput = centered
            previousOutput = highPassed
            output[i] = highPassed
            energy += highPassed * highPassed
            i++
        }
        return sqrt(energy / input.size)
    }
}
