package com.nicos.pitchkit.tuner.harmony.crema

import kotlin.math.ln
import kotlin.math.max

internal data class CremaDecodedChord(
    val label: String,
    val rawLabel: String,
    val confidence: Double,
)

/** Live decoder that fuses Crema's chord tag, root and pitch-content heads. */
internal class CremaHarmonyDecoder(
    private val state: CremaRuntimeState,
    private val preferFlats: Boolean = false,
) {
    private val pitches = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    private val flats = mapOf(1 to "Db", 3 to "Eb", 6 to "Gb", 8 to "Ab", 10 to "Bb")
    private val qualityIntervals = mapOf(
        "7" to intArrayOf(0, 4, 7, 10),
        "aug" to intArrayOf(0, 4, 8),
        "dim" to intArrayOf(0, 3, 6),
        "dim7" to intArrayOf(0, 3, 6, 9),
        "hdim7" to intArrayOf(0, 3, 6, 10),
        "maj" to intArrayOf(0, 4, 7),
        "maj6" to intArrayOf(0, 4, 7, 9),
        "maj7" to intArrayOf(0, 4, 7, 11),
        "min" to intArrayOf(0, 3, 7),
        "min6" to intArrayOf(0, 3, 7, 9),
        "min7" to intArrayOf(0, 3, 7, 10),
        "minmaj7" to intArrayOf(0, 3, 7, 11),
        "sus2" to intArrayOf(0, 2, 7),
        "sus4" to intArrayOf(0, 5, 7),
    )

    fun decode(heads: CremaHeads, smoothingFrames: Int = 3): CremaDecodedChord? {
        val frameCount = heads.frames
        if (frameCount <= 0) return null
        val count = smoothingFrames.coerceIn(1, frameCount)
        val first = frameCount - count

        val tag = average(heads.tag, CremaContract.CHORD_COUNT, first, frameCount)
        val pitch = average(heads.pitch, CremaContract.PITCH_COUNT, first, frameCount)
        val root = average(heads.root, CremaContract.ROOT_COUNT, first, frameCount)
        val bass = average(heads.bass, CremaContract.BASS_COUNT, first, frameCount)

        var noChordProbability = 0.0
        for (index in state.labels.indices) {
            if (state.labels[index] == "N" || state.labels[index] == "X") {
                noChordProbability = max(noChordProbability, tag[index].toDouble().coerceIn(0.0, 1.0))
            }
        }

        var bestIndex = -1
        var bestScore = Double.NEGATIVE_INFINITY
        for (index in state.labels.indices) {
            val label = state.labels[index]
            val parsed = parse(label) ?: continue
            val (rootPc, quality) = parsed
            val intervals = qualityIntervals[quality] ?: continue

            val tagProbability = tag[index].coerceIn(1e-7f, 1.0f)
            val rootProbability = root[rootPc].coerceIn(1e-7f, 1.0f)
            val tones = intervals.map { (rootPc + it) % 12 }.toSet()
            val meanRequired = tones.map { pitch[it].toDouble() }.average().coerceIn(1e-7, 1.0)
            val strongestOutside = pitch.indices
                .filter { it !in tones }
                .maxOfOrNull { pitch[it].toDouble() }
                ?.coerceIn(0.0, 1.0)
                ?: 0.0
            val pitchFit = (0.80 * meanRequired + 0.20 * (1.0 - strongestOutside))
                .coerceIn(1e-7, 1.0)

            val score = 0.70 * ln(tagProbability.toDouble()) +
                0.15 * ln(rootProbability.toDouble()) +
                0.15 * ln(pitchFit)

            if (score > bestScore) {
                bestScore = score
                bestIndex = index
            }
        }

        if (bestIndex < 0) return null
        val raw = state.labels[bestIndex]
        val parsed = parse(raw) ?: return null
        val (rootPc, quality) = parsed
        val chordTones = qualityIntervals[quality] ?: return null
        val chordTagProbability = tag[bestIndex].toDouble().coerceIn(0.0, 1.0)

        if (noChordProbability >= 0.20 && noChordProbability > chordTagProbability) {
            return null
        }

        val bassPc = bass.indices.take(12).maxByOrNull { bass[it] } ?: rootPc
        val relativeBass = (bassPc - rootPc + 12) % 12
        val validInversion = relativeBass != 0 && chordTones.contains(relativeBass)

        val label = buildString {
            append(noteName(rootPc))
            append(displayQuality(quality))
            if (validInversion) {
                append('/')
                append(noteName(bassPc))
            }
        }

        return CremaDecodedChord(
            label = label,
            rawLabel = raw,
            confidence = chordTagProbability,
        )
    }

    private fun average(values: FloatArray, width: Int, first: Int, end: Int): FloatArray {
        val result = FloatArray(width)
        val count = max(1, end - first)
        for (frame in first until end) {
            val offset = frame * width
            for (index in 0 until width) result[index] += values[offset + index]
        }
        for (index in result.indices) result[index] /= count.toFloat()
        return result
    }

    private fun parse(label: String): Pair<Int, String>? {
        if (label == "N" || label == "X") return null
        val separator = label.indexOf(':')
        if (separator <= 0) return null
        val rootName = label.substring(0, separator)
        val quality = label.substring(separator + 1)
        val rootPc = pitches.indexOf(rootName)
        if (rootPc < 0 || quality !in qualityIntervals) return null
        return rootPc to quality
    }

    private fun noteName(pitchClass: Int): String =
        if (preferFlats) flats[pitchClass] ?: pitches[pitchClass] else pitches[pitchClass]

    private fun displayQuality(quality: String): String = when (quality) {
        "maj" -> ""
        "min" -> "m"
        "dim" -> "dim"
        "aug" -> "aug"
        "min6" -> "m6"
        "maj6" -> "6"
        "min7" -> "m7"
        "minmaj7" -> "m(maj7)"
        "maj7" -> "maj7"
        "7" -> "7"
        "dim7" -> "dim7"
        "hdim7" -> "ø7"
        "sus2" -> "sus2"
        "sus4" -> "sus4"
        else -> ":$quality"
    }
}
