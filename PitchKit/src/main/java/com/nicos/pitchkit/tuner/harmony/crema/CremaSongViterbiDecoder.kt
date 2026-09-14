package com.nicos.pitchkit.tuner.harmony.crema

import kotlin.math.ln

/** Whole-song Viterbi decoder for Crema's chord-tag output. */
internal class CremaSongViterbiDecoder(
    private val state: CremaRuntimeState,
    private val preferFlats: Boolean,
) {
    private data class Observation(
        val frame: Long,
        val tag: FloatArray,
        val bass: FloatArray,
    )

    data class Prediction(
        val frame: Long,
        val label: String?,
        val confidence: Double,
    )

    private val observations = mutableListOf<Observation>()

    fun add(heads: CremaHeads, localFrame: Int, globalFrame: Long) {
        require(localFrame in 0 until heads.frames)
        observations += Observation(
            frame = globalFrame,
            tag = heads.tag.copyRow(localFrame, CremaContract.CHORD_COUNT),
            bass = heads.bass.copyRow(localFrame, CremaContract.BASS_COUNT),
        )
    }

    fun decode(): List<Prediction> {
        if (observations.isEmpty()) return emptyList()

        val classCount = state.labels.size
        val timeCount = observations.size
        val logStay = ln(state.transitionDiagonal.coerceAtLeast(EPSILON))
        val logSwitch = ln(state.transitionOffDiagonal.coerceAtLeast(EPSILON))
        val uniformPrior = -ln(classCount.toDouble())

        var previous = DoubleArray(classCount) { stateIndex ->
            uniformPrior + logEmission(observations[0].tag[stateIndex])
        }
        val backPointers = Array(timeCount) { IntArray(classCount) { -1 } }

        for (time in 1 until timeCount) {
            var bestIndex = -1
            var secondIndex = -1
            var bestScore = Double.NEGATIVE_INFINITY
            var secondScore = Double.NEGATIVE_INFINITY
            for (stateIndex in 0 until classCount) {
                val score = previous[stateIndex]
                if (score > bestScore) {
                    secondScore = bestScore
                    secondIndex = bestIndex
                    bestScore = score
                    bestIndex = stateIndex
                } else if (score > secondScore) {
                    secondScore = score
                    secondIndex = stateIndex
                }
            }

            val current = DoubleArray(classCount)
            val emissions = observations[time].tag
            for (stateIndex in 0 until classCount) {
                val stayScore = previous[stateIndex] + logStay
                val switchFrom = if (bestIndex != stateIndex) bestIndex else secondIndex
                val switchBase = if (bestIndex != stateIndex) bestScore else secondScore
                val switchScore = switchBase + logSwitch

                if (stayScore >= switchScore || switchFrom < 0) {
                    current[stateIndex] = stayScore + logEmission(emissions[stateIndex])
                    backPointers[time][stateIndex] = stateIndex
                } else {
                    current[stateIndex] = switchScore + logEmission(emissions[stateIndex])
                    backPointers[time][stateIndex] = switchFrom
                }
            }
            previous = current
        }

        val path = IntArray(timeCount)
        path[timeCount - 1] = previous.indices.maxByOrNull { previous[it] } ?: 0
        for (time in timeCount - 1 downTo 1) {
            val predecessor = backPointers[time][path[time]]
            path[time - 1] = if (predecessor >= 0) predecessor else path[time]
        }

        return List(timeCount) { time ->
            val stateIndex = path[time]
            Prediction(
                frame = observations[time].frame,
                label = displayLabel(state.labels[stateIndex], observations[time].bass),
                confidence = observations[time].tag[stateIndex].toDouble().coerceIn(0.0, 1.0),
            )
        }
    }

    private fun displayLabel(raw: String, bass: FloatArray): String? {
        if (raw == "N" || raw == "X") return null
        val separator = raw.indexOf(':')
        if (separator <= 0) return null
        val rootText = raw.substring(0, separator)
        val quality = raw.substring(separator + 1)
        val rootPc = SHARP_NOTES.indexOf(rootText)
        val chordTones = QUALITY_INTERVALS[quality] ?: return null
        if (rootPc < 0) return null

        val bassPc = (0 until minOf(12, bass.size)).maxByOrNull { bass[it] } ?: rootPc
        val relativeBass = (bassPc - rootPc + 12) % 12
        val validInversion = relativeBass != 0 && chordTones.contains(relativeBass)

        return buildString {
            append(noteName(rootPc))
            append(displayQuality(quality))
            if (validInversion) {
                append('/')
                append(noteName(bassPc))
            }
        }
    }

    private fun noteName(pitchClass: Int): String =
        if (preferFlats) FLAT_NOTES[pitchClass] ?: SHARP_NOTES[pitchClass] else SHARP_NOTES[pitchClass]

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

    private fun logEmission(value: Float): Double = ln(value.toDouble().coerceIn(EPSILON, 1.0))

    private fun FloatArray.copyRow(row: Int, width: Int): FloatArray {
        val offset = row * width
        require(offset >= 0 && offset + width <= size)
        return copyOfRange(offset, offset + width)
    }

    private companion object {
        const val EPSILON = 1e-9
        val SHARP_NOTES = listOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        val FLAT_NOTES = mapOf(1 to "Db", 3 to "Eb", 6 to "Gb", 8 to "Ab", 10 to "Bb")
        val QUALITY_INTERVALS = mapOf(
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
    }
}
