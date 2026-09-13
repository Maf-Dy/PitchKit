package com.nicos.pitchkit.tuner.harmony

import com.nicos.pitchkit.tuner.models.AudioFrame

data class ChordRecognition(
    val label: String,
    val confidence: Double,
    val backend: String,
)

/** Pluggable harmony engine. Implementations may be DSP-only or neural. */
interface ChordRecognizer : AutoCloseable {
    fun recognize(frame: AudioFrame): ChordRecognition?
    fun reset()
    override fun close() = Unit
}
