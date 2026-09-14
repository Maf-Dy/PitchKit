package com.nicos.pitchkit.tuner.harmony.crema

import com.nicos.pitchkit.BuildConfig
import kotlin.math.max

internal data class CremaCandidateDiagnostic(
    val label: String,
    val rawLabel: String,
    val fit: Double,
    val tagConfidence: Double,
)

internal data class CremaDecodedChord(
    val label: String,
    val rawLabel: String,
    val confidence: Double,
    val alternatives: List<CremaCandidateDiagnostic> = emptyList(),
)

/**
 * Low-latency Crema decoder.
 *
 * Crema's reference chord prediction is driven by the chord-tag head. Root and
 * pitch-content heads are auxiliary structured outputs; treating them as extra
 * class-voting terms can move a correct tag prediction to another chord. Keep
 * the model's tag decision authoritative here. Temporal pitch evidence may make
 * a conservative extension/root correction later in the streaming recognizer.
 */
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
        val bass = average(heads.bass, CremaContract.BASS_COUNT, first, frameCount)

        var noChordProbability = 0.0
        var bestIndex = -1
        var bestProbability = Double.NEGATIVE_INFINITY
        val diagnostics = if (BuildConfig.DEBUG) mutableListOf<CremaCandidateDiagnostic>() else null

        for (index in state.labels.indices) {
            val raw = state.labels[index]
            val probability = tag[index].toDouble().coerceIn(0.0, 1.0)
            if (raw == "N" || raw == "X") {
                noChordProbability = max(noChordProbability, probability)
                continue
            }

            val parsed = parse(raw) ?: continue
            if (probability > bestProbability) {
                bestProbability = probability
                bestIndex = index
            }
            diagnostics?.add(
                CremaCandidateDiagnostic(
                    label = noteName(parsed.first) + displayQuality(parsed.second),
                    rawLabel = raw,
                    fit = probability,
                    tagConfidence = probability,
                )
            )
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

        val modelLabel = buildString {
            append(noteName(rootPc))
            append(displayQuality(quality))
            if (validInversion) {
                append('/')
                append(noteName(bassPc))
            }
        }

        return CremaDecodedChord(
            label = modelLabel,
            rawLabel = raw,
            confidence = chordTagProbability,
            alternatives = diagnostics
                ?.sortedByDescending { it.tagConfidence }
                ?.take(3)
                .orEmpty(),
        )
    }

    private fun average(values: FloatArray, width: Int, first: Int, end: Int): FloatArray {
        val result = FloatArray(width)
        val count = max(1, end - first)
        for (frame in first until end) {
            val offset = frame * width
            for (index in result.indices) result[index] += values[offset + index]
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
