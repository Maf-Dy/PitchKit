package com.nicos.pitchkit.tuner

/** Public result emitted by [PitchAnalyzer] and the microphone convenience listener. */
sealed class TuningResult {
    data class Note(
        val name: String,
        val cents: Double,
        val freq: Float,
        /** [name] with its octave, e.g. `"E2"`; shows octave errors and matches open strings. */
        val nameWithOctave: String = name,
    ) : TuningResult()

    /**
     * A chord that passed the active detector's own acceptance rules.
     *
     * [confidence] is detector-specific and is primarily diagnostic; callers should
     * not compare values from different [backend] implementations as if they were
     * calibrated to the same probability scale.
     *
     * [backend] carries the negotiated audio source as a suffix when the live
     * engine knows it, e.g. `"Crema 0.2.0 · unprocessed"`.
     *
     * [droppedFrames] is how many capture buffers the conflated capture channel
     * has discarded since the engine started — a debug counter, and zero for
     * callers that build a result themselves.
     */
    data class Chord(
        val name: String,
        val confidence: Double = 1.0,
        val backend: String = "Unknown",
        val droppedFrames: Long = 0L,
    ) : TuningResult()

    object Silence : TuningResult()
}
