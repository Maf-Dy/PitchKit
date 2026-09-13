package com.nicos.pitchkit.tuner.harmony.chordnet

internal class FloatRingBuffer(
    capacity: Int,
) {
    private val values = FloatArray(capacity)
    private var writeIndex = 0
    var size: Int = 0
        private set

    init {
        require(capacity > 0)
    }

    fun append(samples: FloatArray) {
        for (sample in samples) {
            values[writeIndex] = sample
            writeIndex = (writeIndex + 1) % values.size
            if (size < values.size) size++
        }
    }

    fun toFloatArray(): FloatArray {
        val result = FloatArray(size)
        val start = (writeIndex - size + values.size) % values.size
        for (index in 0 until size) {
            result[index] = values[(start + index) % values.size]
        }
        return result
    }

    fun clear() {
        writeIndex = 0
        size = 0
    }
}
