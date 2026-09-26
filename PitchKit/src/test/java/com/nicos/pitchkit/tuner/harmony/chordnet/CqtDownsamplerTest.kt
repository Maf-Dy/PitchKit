package com.nicos.pitchkit.tuner.harmony.chordnet

import java.io.File
import kotlin.random.Random
import org.junit.Assert.assertArrayEquals
import org.junit.Test

class CqtDownsamplerTest {
    @Test fun shippedPlansRemainBitExactThroughRecursiveDecimation() {
        val assets = sequenceOf(File("src/main/assets"), File("PitchKit/PitchKit/src/main/assets"))
            .first { File(it, "crema/crema-cqt-h1.bin").isFile }
        val random = Random(73517)
        for (name in listOf("crema/crema-cqt-h1.bin", "crema/crema-cqt-h2.bin", "btc/cqt-plan.bin", "chordnet/cqt-plan.bin")) {
            val plan = CqtPlanDecoder.decodeAndVerify(File(assets, name).readBytes()).downsample
            val optimized = CqtDownsampler(plan)
            for (size in listOf(0, 1, 2, 3, 127, 254, 255, 256, 511, 4097, 98304)) {
                var actual = FloatArray(size) { random.nextFloat() * 2f - 1f }
                var expected = actual.copyOf()
                repeat(8) { level ->
                    expected = reference(expected, plan)
                    actual = optimized.transform(actual)
                    assertArrayEquals("$name size=$size level=$level",
                        expected.map { it.toRawBits() }.toIntArray(), actual.map { it.toRawBits() }.toIntArray())
                }
            }
        }
    }

    @Test fun silenceAndEdgeImpulsesKeepAlignmentGainAndOutputLength() {
        val plan = CqtDownsamplePlan(5, floatArrayOf(0.5f, 0.25f, -0.125f))
        val optimized = CqtDownsampler(plan)
        for (size in 1..19) {
            for (position in -1 until size) {
                val input = FloatArray(size) { if (it == position) 1f else 0f }
                assertArrayEquals(reference(input, plan), optimized.transform(input), 0f)
            }
        }
    }

    // Frozen pre-optimization implementation: independent bounds/mirror handling.
    private fun reference(input: FloatArray, plan: CqtDownsamplePlan): FloatArray {
        val output = FloatArray(kotlin.math.ceil(input.size / 2.0).toInt())
        for (index in output.indices) {
            var value = 0.0
            for (tap in 0 until plan.tapCount) {
                val source = index * 2 + tap - plan.delay
                if (source !in input.indices) continue
                value += input[source] * plan.halfCoefficients[kotlin.math.abs(tap - plan.delay)]
            }
            output[index] = (value * plan.gain).toFloat()
        }
        return output
    }
}
