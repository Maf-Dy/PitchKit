package com.nicos.pitchkit.tuner

/** Public result emitted by [PitchAnalyzer] and the microphone convenience listener. */
sealed class TuningResult {
    data class Note(
        val name: String,
        val cents: Double,
        val freq: Float,
    ) : TuningResult()

    /**
     * A chord that passed the active detector's own acceptance rules.
     *
     * [confidence] is detector-specific and is primarily diagnostic; callers should
     * not compare values from different [backend] implementations as if they were
     * calibrated to the same probability scale.
     */
    data class Chord(
        val name: String,
        val confidence: Double = 1.0,
        val backend: String = "Unknown",
    ) : TuningResult()

    object Silence : TuningResult()
}
