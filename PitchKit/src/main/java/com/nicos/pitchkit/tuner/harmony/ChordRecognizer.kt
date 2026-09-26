package com.nicos.pitchkit.tuner.harmony

import com.nicos.pitchkit.tuner.models.AudioFrame

data class ChordRecognition(
    val label: String,
    val confidence: Double,
    val backend: String,
)

/** Pluggable harmony engine. Implementations may be DSP-only or neural. */
interface ChordRecognizer : AutoCloseable {
    /** New object for each fresh decision, including abstention; unchanged for cached/no-update calls. */
    val latestUpdate: ChordRecognitionUpdate? get() = null
    fun recognize(frame: AudioFrame): ChordRecognition?
    fun reset()
    override fun close() = Unit
}

data class ChordRecognitionUpdate(val recognition: ChordRecognition?)

internal class ChordUpdateTracker {
    var latest: ChordRecognitionUpdate? = null
        private set
    fun record(recognition: ChordRecognition?): ChordRecognition? {
        latest=ChordRecognitionUpdate(recognition)
        return recognition
    }
    fun reset() {latest=null}
}
