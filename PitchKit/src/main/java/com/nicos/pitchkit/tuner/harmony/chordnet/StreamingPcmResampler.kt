package com.nicos.pitchkit.tuner.harmony.chordnet

import kotlin.math.PI
import kotlin.math.abs

/** Stateful resampler for arbitrary PCM sources feeding the 22.05 kHz model. */
internal class StreamingPcmResampler(
    private val targetRate: Int = ChordNetContract.SAMPLE_RATE,
    private val pitchScale: Double = 1.0,
) {
    private var sourceRate = -1
    private var sourcePerOutput = 1.0
    private var position = 0.0
    private var hasPrevious = false
    private var previous = 0f

    private val lowPassState = FloatArray(4)
    private var lowPassAlpha = 1f

    init {
        require(targetRate > 0)
        require(pitchScale > 0.0)
    }

    fun process(input: FloatArray, inputRate: Int): FloatArray {
        if (input.isEmpty()) return FloatArray(0)
        require(inputRate > 0)
        if (inputRate != sourceRate) configure(inputRate)

        val filtered = if (sourcePerOutput > 1.0) lowPass(input) else input
        if (inputRate == targetRate && abs(pitchScale - 1.0) < 1e-9) {
            previous = filtered.last()
            hasPrevious = true
            return filtered.copyOf()
        }

        val combined = if (hasPrevious) {
            FloatArray(filtered.size + 1).also { values ->
                values[0] = previous
                filtered.copyInto(values, destinationOffset = 1)
            }
        } else {
            filtered
        }

        if (combined.size < 2) {
            previous = combined.last()
            hasPrevious = true
            return FloatArray(0)
        }

        val output = ArrayList<Float>((combined.size / sourcePerOutput + 2).toInt())
        var cursor = position
        while (cursor < combined.lastIndex) {
            val left = cursor.toInt().coerceIn(0, combined.lastIndex - 1)
            val fraction = (cursor - left).toFloat()
            output += combined[left] + (combined[left + 1] - combined[left]) * fraction
            cursor += sourcePerOutput
        }

        cursor -= combined.lastIndex.toDouble()
        position = cursor.coerceAtLeast(0.0)
        previous = combined.last()
        hasPrevious = true
        return FloatArray(output.size) { output[it] }
    }

    fun reset() {
        sourceRate = -1
        sourcePerOutput = 1.0
        position = 0.0
        hasPrevious = false
        previous = 0f
        lowPassState.fill(0f)
    }

    private fun configure(inputRate: Int) {
        reset()
        sourceRate = inputRate
        sourcePerOutput = inputRate.toDouble() / targetRate * pitchScale

        if (sourcePerOutput > 1.0) {
            val cutoffHz = (targetRate * 0.40 / pitchScale).coerceAtMost(inputRate * 0.45)
            val dt = 1.0 / inputRate
            val rc = 1.0 / (2.0 * PI * cutoffHz)
            lowPassAlpha = (dt / (rc + dt)).toFloat()
        } else {
            lowPassAlpha = 1f
        }
    }

    private fun lowPass(input: FloatArray): FloatArray {
        val output = FloatArray(input.size)
        for (index in input.indices) {
            var value = input[index]
            for (stage in lowPassState.indices) {
                val next = lowPassState[stage] + lowPassAlpha * (value - lowPassState[stage])
                lowPassState[stage] = next
                value = next
            }
            output[index] = value
        }
        return output
    }
}
