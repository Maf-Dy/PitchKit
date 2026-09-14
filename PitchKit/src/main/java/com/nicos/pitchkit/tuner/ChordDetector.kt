package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Polyphonic chord detector built around note-fundamental salience rather than
 * directly folding every FFT peak into chroma.
 *
 * Classic deliberately stays an independent-frame DSP fallback. Richer
 * extensions are supported, but they must have direct spectral evidence so the
 * detector does not invent jazz extensions from passing harmonics.
 */
internal class ChordDetector(
    private val sampleRate: Int,
    private val profile: InstrumentProfile,
    private val referenceA4Hz: Double = 440.0,
) {
    private data class Template(
        val name: String,
        val suffix: String,
        val pitches: IntArray,
        val pitchMask: Int,
        val coreSize: Int = 3,
    )

    private data class PitchProfile(
        val chroma: DoubleArray,
        val bassPitchClass: Int,
    )

    data class ChordResult(
        val name: String,
        val score: Double,
        val confidence: Double,
    )

    private val templates = buildTemplates()
    private val maxFundamental = min(
        profile.maxFreq,
        max(profile.bassCeiling * 3.0, profile.harmonicPivot * 6.0),
    )
    private val minMidi = floor(midiForFrequency(profile.minFreq)).toInt() - 1
    private val maxMidi = ceil(midiForFrequency(maxFundamental)).toInt() + 1

    private val topPitchClassScore = DoubleArray(12)
    private val secondPitchClassScore = DoubleArray(12)
    private val chromaScratch = DoubleArray(12)
    private val noteMidiScratch = IntArray((maxMidi - minMidi + 1).coerceAtLeast(1))
    private val noteScoreScratch = DoubleArray(noteMidiScratch.size)

    private var currentChord: String? = null
    private var pendingChord: String? = null
    private var pendingCount = 0

    init {
        require(referenceA4Hz > 0.0) { "referenceA4Hz must be > 0" }
    }

    fun detect(
        buffer: FloatArray,
        minScore: Double = 0.20,
        minMargin: Double = 0.035,
    ): ChordResult? {
        val profileNow = buildPitchProfile(buffer)
        val chroma = profileNow.chroma
        val bass = profileNow.bassPitchClass

        var bestTemplate: Template? = null
        var bestScore = Double.NEGATIVE_INFINITY
        var secondScore = Double.NEGATIVE_INFINITY

        for (template in templates) {
            val score = scoreTemplate(template, chroma, bass)
            if (score > bestScore) {
                secondScore = bestScore
                bestScore = score
                bestTemplate = template
            } else if (score > secondScore) {
                secondScore = score
            }
        }

        val candidate = bestTemplate?.name ?: return null
        val margin = if (secondScore.isFinite()) bestScore - secondScore else bestScore

        if (bestScore < minScore || margin < minMargin) {
            clearPending()
            return null
        }

        if (candidate == currentChord) {
            clearPending()
            return ChordResult(candidate, bestScore, margin.coerceAtLeast(0.0))
        }

        if (candidate == pendingChord) {
            pendingCount++
        } else {
            pendingChord = candidate
            pendingCount = 1
        }

        if (pendingCount < 2) return null

        currentChord = candidate
        clearPending()
        return ChordResult(candidate, bestScore, margin.coerceAtLeast(0.0))
    }

    fun chroma(buffer: FloatArray): DoubleArray = buildPitchProfile(buffer).chroma

    fun reset() {
        currentChord = null
        clearPending()
    }

    private fun buildPitchProfile(buffer: FloatArray): PitchProfile {
        val magnitudes = FFT.magnitudePadded(buffer, padFactor = 4)
        val fftSize = magnitudes.size * 2

        java.util.Arrays.fill(topPitchClassScore, 0.0)
        java.util.Arrays.fill(secondPitchClassScore, 0.0)
        java.util.Arrays.fill(chromaScratch, 0.0)

        var noteCount = 0
        var strongestNote = 0.0

        for (midi in minMidi..maxMidi) {
            val frequency = frequencyForMidi(midi)
            if (frequency < profile.minFreq || frequency > maxFundamental) continue

            val salience = fundamentalSalience(
                magnitudes = magnitudes,
                fftSize = fftSize,
                fundamentalHz = frequency,
            )
            if (salience <= 0.0) continue

            val pitchClass = floorMod12(midi)
            if (salience > topPitchClassScore[pitchClass]) {
                secondPitchClassScore[pitchClass] = topPitchClassScore[pitchClass]
                topPitchClassScore[pitchClass] = salience
            } else if (salience > secondPitchClassScore[pitchClass]) {
                secondPitchClassScore[pitchClass] = salience
            }

            if (noteCount < noteScoreScratch.size) {
                noteMidiScratch[noteCount] = midi
                noteScoreScratch[noteCount] = salience
                noteCount++
            }
            if (salience > strongestNote) strongestNote = salience
        }

        var peak = 0.0
        for (pitchClass in chromaScratch.indices) {
            val value = topPitchClassScore[pitchClass] + secondPitchClassScore[pitchClass]
            chromaScratch[pitchClass] = value
            if (value > peak) peak = value
        }
        if (peak > 0.0) {
            for (i in chromaScratch.indices) chromaScratch[i] /= peak
        }

        var bassMidi = Int.MAX_VALUE
        if (strongestNote > 0.0) {
            val threshold = strongestNote * 0.30
            for (index in 0 until noteCount) {
                if (noteScoreScratch[index] >= threshold && noteMidiScratch[index] < bassMidi) {
                    bassMidi = noteMidiScratch[index]
                }
            }
        }

        return PitchProfile(
            chroma = chromaScratch,
            bassPitchClass = if (bassMidi == Int.MAX_VALUE) -1 else floorMod12(bassMidi),
        )
    }

    private fun fundamentalSalience(
        magnitudes: DoubleArray,
        fftSize: Int,
        fundamentalHz: Double,
    ): Double {
        val direct = compressedPeakNear(magnitudes, fftSize, fundamentalHz)
        if (direct <= 0.0) return 0.0

        val harmonicWeights = doubleArrayOf(0.0, 0.0, 0.34, 0.22, 0.14, 0.10, 0.08)
        var support = 0.0
        for (harmonic in 2..6) {
            val harmonicHz = fundamentalHz * harmonic
            if (harmonicHz > profile.maxFreq) break
            support += harmonicWeights[harmonic] * compressedPeakNear(magnitudes, fftSize, harmonicHz)
        }

        var leakage = 0.0
        val oddHarmonics = intArrayOf(3, 5, 7)
        val leakageWeights = doubleArrayOf(0.24, 0.16, 0.10)
        for (index in oddHarmonics.indices) {
            val possibleFundamental = fundamentalHz / oddHarmonics[index]
            if (possibleFundamental >= profile.minFreq) {
                leakage += leakageWeights[index] *
                    compressedPeakNear(magnitudes, fftSize, possibleFundamental)
            }
        }

        return max(0.0, direct + support - leakage)
    }

    private fun compressedPeakNear(
        magnitudes: DoubleArray,
        fftSize: Int,
        frequencyHz: Double,
        radiusCents: Double = 35.0,
    ): Double {
        if (frequencyHz <= 0.0 || frequencyHz >= sampleRate / 2.0) return 0.0

        val ratio = 2.0.pow(radiusCents / 1200.0)
        val lowHz = frequencyHz / ratio
        val highHz = frequencyHz * ratio
        val lowBin = floor(lowHz * fftSize / sampleRate)
            .toInt()
            .coerceIn(1, magnitudes.lastIndex)
        val highBin = ceil(highHz * fftSize / sampleRate)
            .toInt()
            .coerceIn(lowBin, magnitudes.lastIndex)

        var peak = 0.0
        for (bin in lowBin..highBin) {
            if (magnitudes[bin] > peak) peak = magnitudes[bin]
        }
        return ln(1.0 + peak)
    }

    private fun scoreTemplate(
        template: Template,
        chroma: DoubleArray,
        bass: Int,
    ): Double {
        var toneSum = 0.0
        var weakestTone = Double.POSITIVE_INFINITY
        var outside1 = 0.0
        var outside2 = 0.0
        var outside3 = 0.0
        var outsideCount = 0

        for (pitchClass in chroma.indices) {
            val value = chroma[pitchClass]
            if ((template.pitchMask and (1 shl pitchClass)) != 0) {
                toneSum += value
                if (value < weakestTone) weakestTone = value
            } else {
                outsideCount++
                when {
                    value > outside1 -> {
                        outside3 = outside2
                        outside2 = outside1
                        outside1 = value
                    }
                    value > outside2 -> {
                        outside3 = outside2
                        outside2 = value
                    }
                    value > outside3 -> outside3 = value
                }
            }
        }

        val meanTone = toneSum / template.pitches.size
        if (!weakestTone.isFinite()) weakestTone = 0.0
        val outsideSamples = min(3, outsideCount)
        val strongestOutside = if (outsideSamples == 0) {
            0.0
        } else {
            (outside1 + outside2 + outside3) / outsideSamples
        }

        var score = 0.68 * meanTone +
            0.32 * weakestTone -
            0.34 * strongestOutside

        val root = template.pitches[0]
        score += 0.08 * chroma[root]

        if (bass >= 0) {
            val bassIsTone = (template.pitchMask and (1 shl bass)) != 0
            score += when {
                bass == root -> 0.10
                bassIsTone -> 0.025
                else -> -0.07
            }
        }

        val extensionCount = (template.pitches.size - template.coreSize).coerceAtLeast(0)
        if (extensionCount > 0) {
            var coreMean = 0.0
            for (index in 0 until template.coreSize.coerceAtMost(template.pitches.size)) {
                coreMean += chroma[template.pitches[index]]
            }
            coreMean /= template.coreSize.coerceAtMost(template.pitches.size).coerceAtLeast(1)
            val extensionThreshold = max(0.18, coreMean * 0.28)
            for (index in template.coreSize until template.pitches.size) {
                val support = chroma[template.pitches[index]]
                score += if (support >= extensionThreshold) 0.022 else -0.105
            }
            score -= 0.012 * extensionCount
        }

        if (template.suffix == "sus2" || template.suffix == "sus4") {
            val minorThird = (root + 3) % 12
            val majorThird = (root + 4) % 12
            score -= 0.12 * max(chroma[minorThird], chroma[majorThird])
        }

        if (template.suffix == "" || template.suffix == "m") {
            score += 0.025
        }

        return score
    }

    private fun buildTemplates(): List<Template> {
        val roots = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        val flats = mapOf("C#" to "Db", "D#" to "Eb", "F#" to "Gb", "G#" to "Ab", "A#" to "Bb")
        val qualities = listOf(
            "" to intArrayOf(0, 4, 7),
            "m" to intArrayOf(0, 3, 7),
            "7" to intArrayOf(0, 4, 7, 10),
            "m7" to intArrayOf(0, 3, 7, 10),
            "maj7" to intArrayOf(0, 4, 7, 11),
            "6" to intArrayOf(0, 4, 7, 9),
            "m6" to intArrayOf(0, 3, 7, 9),
            "m(maj7)" to intArrayOf(0, 3, 7, 11),
            "6/9" to intArrayOf(0, 4, 7, 2, 9),
            "9" to intArrayOf(0, 4, 7, 10, 2),
            "maj9" to intArrayOf(0, 4, 7, 11, 2),
            "m9" to intArrayOf(0, 3, 7, 10, 2),
            "11" to intArrayOf(0, 4, 7, 10, 2, 5),
            "m11" to intArrayOf(0, 3, 7, 10, 2, 5),
            "13" to intArrayOf(0, 4, 7, 10, 2, 9),
            "maj13" to intArrayOf(0, 4, 7, 11, 2, 9),
            "m13" to intArrayOf(0, 3, 7, 10, 2, 9),
            "7b9" to intArrayOf(0, 4, 7, 10, 1),
            "7#9" to intArrayOf(0, 4, 7, 10, 3),
            "7#11" to intArrayOf(0, 4, 7, 10, 6),
            "7b13" to intArrayOf(0, 4, 7, 10, 8),
            "9#11" to intArrayOf(0, 4, 7, 10, 2, 6),
            "dim7" to intArrayOf(0, 3, 6, 9),
            "ø7" to intArrayOf(0, 3, 6, 10),
            "sus2" to intArrayOf(0, 2, 7),
            "sus4" to intArrayOf(0, 5, 7),
            "dim" to intArrayOf(0, 3, 6),
            "aug" to intArrayOf(0, 4, 8),
        )

        return buildList {
            for (root in roots.indices) {
                for ((suffix, intervals) in qualities) {
                    val pitches = intervals.map { (it + root) % 12 }.toIntArray()
                    var mask = 0
                    for (pitch in pitches) mask = mask or (1 shl pitch)
                    val displayRoot = if (profile.useFlats) flats[roots[root]] ?: roots[root] else roots[root]
                    add(
                        Template(
                            name = displayRoot + suffix,
                            suffix = suffix,
                            pitches = pitches,
                            pitchMask = mask,
                        )
                    )
                }
            }
        }
    }

    private fun clearPending() {
        pendingChord = null
        pendingCount = 0
    }

    private fun midiForFrequency(frequencyHz: Double): Double =
        69.0 + 12.0 * (ln(frequencyHz / referenceA4Hz) / ln(2.0))

    private fun frequencyForMidi(midi: Int): Double =
        referenceA4Hz * 2.0.pow((midi - 69) / 12.0)

    private fun floorMod12(value: Int): Int = ((value % 12) + 12) % 12
}
