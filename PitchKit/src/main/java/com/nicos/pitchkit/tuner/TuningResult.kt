package com.nicos.pitchkit.tuner
// Modified in Maf-Dy/PitchKit fork: DSP correctness, performance, and lifecycle fixes.

sealed class TuningResult {
    data class Note(
        val name: String,
        val cents: Double,
        val freq: Float,
    ) : TuningResult()

    data class Chord(
        val name: String,
        val score: Double = 0.0,
        val confidence: Double = 0.0,
    ) : TuningResult()

    object Silence : TuningResult()
}
