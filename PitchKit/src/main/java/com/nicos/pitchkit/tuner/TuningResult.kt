package com.nicos.pitchkit.tuner

/** Public result emitted by [PitchAnalyzer] and the microphone convenience listener. */
sealed class TuningResult {
    /**
     * A single detected note.
     * @param name note name, e.g. "E" or "A#".
     * @param cents deviation from the configured reference tuning; 0 = in tune.
     * @param freq detected frequency in Hz.
     */
    data class Note(
        val name: String,
        val cents: Double,
        val freq: Float,
    ) : TuningResult()

    /** A detected chord, e.g. "Am" or "Cmaj7". */
    data class Chord(val name: String) : TuningResult()

    /** No usable signal / below the detection threshold. */
    object Silence : TuningResult()
}
