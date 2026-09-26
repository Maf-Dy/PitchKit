package com.nicos.pitchkit.tuner.harmony.chordnet

/** Plan FIR decimation with clipped edges and pre-expanded coefficients. */
internal class CqtDownsampler(private val plan: CqtDownsamplePlan) {
    // Avoid recalculating the mirror index for every tap of every output sample.
    private val coefficients = FloatArray(plan.tapCount) { tap ->
        plan.halfCoefficients[kotlin.math.abs(tap - plan.delay)]
    }

    fun transform(input: FloatArray): FloatArray {
        val output = FloatArray(input.size / 2 + input.size % 2)
        for (outputIndex in output.indices) {
            val start = outputIndex * 2 - plan.delay
            var tap = maxOf(0, -start)
            val end = minOf(coefficients.size, input.size - start)
            var source = start + tap
            var value = 0.0
            // Keep the original increasing-tap accumulation and FLOAT product.
            // Pairing symmetric samples or promoting the product changes rounding.
            while (tap < end) {
                value += input[source] * coefficients[tap]
                source++
                tap++
            }
            output[outputIndex] = (value * plan.gain).toFloat()
        }
        return output
    }
}
