package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

internal class ChordDetector(
    private val sampleRate: Int,
    private val profile: InstrumentProfile,
    private val referenceA4Hz: Double = 440.0,
) {
    private data class Template(val name: String, val pitches: IntArray)

    private val templates = buildTemplates()
    private var currentChord: String? = null
    private var currentScore = 0.0
    private var pendingChord: String? = null
    private var pendingCount = 0
    private val chromaHistory = ArrayDeque<DoubleArray>()
    private val chromaWindow = 3

    init {
        require(referenceA4Hz > 0.0) { "referenceA4Hz must be > 0" }
    }

    private fun buildTemplates(): List<Template> {
        val roots = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        val flat = mapOf("C#" to "Db", "D#" to "Eb", "F#" to "Gb", "G#" to "Ab", "A#" to "Bb")
        val qualities = listOf(
            "" to intArrayOf(0, 4, 7),
            "m" to intArrayOf(0, 3, 7),
            "7" to intArrayOf(0, 4, 7, 10),
            "m7" to intArrayOf(0, 3, 7, 10),
            "maj7" to intArrayOf(0, 4, 7, 11),
            "sus2" to intArrayOf(0, 2, 7),
            "sus4" to intArrayOf(0, 5, 7),
            "dim" to intArrayOf(0, 3, 6),
            "aug" to intArrayOf(0, 4, 8),
        )

        return buildList {
            for (root in roots.indices) {
                for ((suffix, intervals) in qualities) {
                    val rootName = if (profile.useFlats) flat[roots[root]] ?: roots[root] else roots[root]
                    add(Template(rootName + suffix, intervals.map { (it + root) % 12 }.toIntArray()))
                }
            }
        }
    }

    private fun chromaFrom(mags: DoubleArray, n: Int): DoubleArray {
        val chroma = DoubleArray(12)
        for (bin in 1 until mags.size - 1) {
            if (mags[bin] < mags[bin - 1] || mags[bin] < mags[bin + 1]) continue

            val offset = FFT.interpolatePeak(mags, bin)
            val freq = (bin + offset) * sampleRate.toDouble() / n
            if (freq < profile.minFreq || freq > profile.maxFreq) continue

            val weight = sqrt(profile.harmonicPivot / max(freq, profile.harmonicPivot))
            val midi = midiForFrequency(freq)
            val pitchClass = ((midi.roundToInt() % 12) + 12) % 12
            chroma[pitchClass] += mags[bin] * weight
        }

        val peak = chroma.maxOrNull() ?: 1.0
        if (peak > 0.0) {
            for (i in chroma.indices) chroma[i] /= peak
        }
        return chroma
    }

    private fun smoothedChroma(current: DoubleArray): DoubleArray {
        chromaHistory.addLast(current)
        if (chromaHistory.size > chromaWindow) chromaHistory.removeFirst()

        val average = DoubleArray(12)
        for (frame in chromaHistory) {
            for (i in average.indices) average[i] += frame[i]
        }
        for (i in average.indices) average[i] /= chromaHistory.size
        return average
    }

    private fun detectBassPitchClass(mags: DoubleArray, n: Int): Int {
        val maxMagnitude = mags.maxOrNull() ?: return -1
        for (bin in 1 until mags.size) {
            val freq = bin * sampleRate.toDouble() / n
            if (freq < profile.minFreq) continue
            if (freq > profile.bassCeiling) break
            if (mags[bin] > maxMagnitude * 0.3) {
                val midi = midiForFrequency(freq)
                return ((midi.roundToInt() % 12) + 12) % 12
            }
        }
        return -1
    }

    data class ChordResult(val name: String, val score: Double)

    fun detect(buffer: FloatArray, minScore: Double = 0.20): ChordResult? {
        val mags = FFT.magnitudePadded(buffer, padFactor = 2)
        val n = mags.size * 2
        val chroma = smoothedChroma(chromaFrom(mags, n))
        if (chroma.sum() < 0.5) {
            clearPending()
            return null
        }

        val bass = detectBassPitchClass(mags, n)
        var best: Template? = null
        var bestScore = -1.0

        for (template in templates) {
            val tones = template.pitches.toHashSet()
            var inChord = 0.0
            var outChord = 0.0
            for (pitchClass in 0 until 12) {
                if (pitchClass in tones) inChord += chroma[pitchClass]
                else outChord += chroma[pitchClass]
            }

            var score = inChord / template.pitches.size -
                0.5 * outChord / (12 - template.pitches.size)
            if (bass >= 0 && template.pitches.isNotEmpty() && template.pitches[0] == bass) {
                score += 0.15
            }
            if (score > bestScore) {
                bestScore = score
                best = template
            }
        }

        if (bestScore < minScore) {
            clearPending()
            return null
        }

        val candidate = best?.name ?: return null
        if (candidate == currentChord) {
            currentScore = bestScore
            clearPending()
            return ChordResult(candidate, bestScore)
        }

        if (candidate == pendingChord) {
            pendingCount++
        } else {
            pendingChord = candidate
            pendingCount = 1
        }

        // First valid chord is immediate. Later changes need two consecutive
        // candidate frames. This prevents flicker without permanently favouring a
        // previously high-scoring chord over a legitimate lower-scoring next one.
        if (currentChord == null || pendingCount >= 2) {
            currentChord = candidate
            currentScore = bestScore
            clearPending()
        }

        return currentChord?.let { ChordResult(it, currentScore) }
    }

    fun reset() {
        currentChord = null
        currentScore = 0.0
        clearPending()
        chromaHistory.clear()
    }

    fun chroma(buffer: FloatArray): DoubleArray {
        val mags = FFT.magnitudePadded(buffer, padFactor = 2)
        return chromaFrom(mags, n = mags.size * 2)
    }

    private fun clearPending() {
        pendingChord = null
        pendingCount = 0
    }

    private fun midiForFrequency(freq: Double): Double {
        return 69 + 12 * (ln(freq / referenceA4Hz) / ln(2.0))
    }
}
