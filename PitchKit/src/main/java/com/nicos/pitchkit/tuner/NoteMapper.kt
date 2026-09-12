package com.nicos.pitchkit.tuner

import kotlin.math.ln
import kotlin.math.roundToInt

internal object NoteMapper {
    private val names = arrayOf(
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    )

    private val flatAliases = mapOf(
        "A#" to "Bb", "C#" to "Db", "D#" to "Eb", "F#" to "Gb", "G#" to "Ab"
    )

    data class NoteResult(
        val name: String,
        val nameWithOctave: String,
        val cents: Double,
        val frequency: Float,
    )

    /** Converts a frequency to the nearest equal-tempered note. */
    fun frequencyToNote(
        freq: Float,
        useFlats: Boolean = false,
        referenceA4Hz: Double = 440.0,
    ): NoteResult? {
        if (freq <= 0f) return null
        require(referenceA4Hz > 0.0) { "referenceA4Hz must be > 0" }

        val midi = 69.0 + 12.0 * log2(freq.toDouble() / referenceA4Hz)
        val nearest = midi.roundToInt()
        val cents = (midi - nearest) * 100.0

        val noteIdx = ((nearest % 12) + 12) % 12
        val octave = nearest / 12 - 1
        var name = names[noteIdx]
        if (useFlats) name = flatAliases[name] ?: name

        return NoteResult(name, "$name$octave", cents, freq)
    }

    private fun log2(value: Double): Double = ln(value) / ln(2.0)
}
