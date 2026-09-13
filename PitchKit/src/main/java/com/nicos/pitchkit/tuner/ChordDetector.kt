package com.nicos.pitchkit.tuner
// Modified in Maf-Dy/PitchKit fork: DSP correctness, performance, and lifecycle fixes.

import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

internal class ChordDetector(
    private val sampleRate: Int,
    private val profile: InstrumentProfile,
) {
    private data class Template(
        val name: String,
        val rootPc: Int,
        val pitches: IntArray,
        val mask: BooleanArray,
    )

    data class ChordResult(
        val name: String,
        val score: Double,
        val confidence: Double,
        val bassPitchClass: Int,
    )

    data class FrameAnalysis(val chroma: DoubleArray, val chord: ChordResult?)

    private val templates = buildTemplates()

    fun detect(buffer: FloatArray, minScore: Double = 0.30): ChordResult? =
        analyze(buffer, minScore).chord

    fun analyze(buffer: FloatArray, minScore: Double = 0.30): FrameAnalysis {
        val mags = FFT.magnitudePadded(buffer, padFactor = 2)
        val n = mags.size * 2
        val chroma = chromaFrom(mags, n)
        if (chroma.sum() < 0.45) return FrameAnalysis(chroma, null)
        val bass = detectBassPitchClass(mags, n)
        return FrameAnalysis(chroma, scoreTemplates(chroma, bass, minScore))
    }

    fun chroma(buffer: FloatArray): DoubleArray {
        val mags = FFT.magnitudePadded(buffer, padFactor = 2)
        return chromaFrom(mags, mags.size * 2)
    }

    fun reset() = Unit

    private fun buildTemplates(): List<Template> {
        val roots = arrayOf("C","C#","D","D#","E","F","F#","G","G#","A","A#","B")
        val flats = mapOf("C#" to "Db","D#" to "Eb","F#" to "Gb","G#" to "Ab","A#" to "Bb")
        val qualities = listOf(
            "" to intArrayOf(0,4,7), "m" to intArrayOf(0,3,7),
            "7" to intArrayOf(0,4,7,10), "m7" to intArrayOf(0,3,7,10),
            "maj7" to intArrayOf(0,4,7,11), "sus2" to intArrayOf(0,2,7),
            "sus4" to intArrayOf(0,5,7), "dim" to intArrayOf(0,3,6),
            "dim7" to intArrayOf(0,3,6,9), "m7b5" to intArrayOf(0,3,6,10),
            "aug" to intArrayOf(0,4,8),
        )
        val out = ArrayList<Template>(roots.size * qualities.size)
        for (root in roots.indices) {
            for ((suffix, intervals) in qualities) {
                val pitches = IntArray(intervals.size) { (intervals[it] + root) % 12 }
                val mask = BooleanArray(12)
                pitches.forEach { mask[it] = true }
                val rootName = if (profile.useFlats) flats[roots[root]] ?: roots[root] else roots[root]
                out += Template(rootName + suffix, root, pitches, mask)
            }
        }
        return out
    }

    private fun chromaFrom(mags: DoubleArray, n: Int): DoubleArray {
        val chroma = DoubleArray(12)
        val maxMag = mags.maxOrNull() ?: return chroma
        if (maxMag <= 0.0) return chroma
        val peakFloor = maxMag * 0.015
        for (bin in 1 until mags.size - 1) {
            val magnitude = mags[bin]
            if (magnitude < peakFloor) continue
            if (magnitude < mags[bin - 1] || magnitude < mags[bin + 1]) continue
            val offset = FFT.interpolatePeak(mags, bin)
            val freq = (bin + offset) * sampleRate.toDouble() / n
            if (freq < profile.minFreq || freq > profile.maxFreq) continue

            val compressedMagnitude = sqrt(magnitude / maxMag)
            val harmonicWeight = sqrt(profile.harmonicPivot / max(freq, profile.harmonicPivot))
            val midi = 69 + 12 * (ln(freq / 440.0) / ln(2.0))
            val pc = ((midi.roundToInt() % 12) + 12) % 12
            chroma[pc] += compressedMagnitude * harmonicWeight
        }
        val peak = chroma.maxOrNull() ?: 0.0
        if (peak > 0.0) for (i in chroma.indices) chroma[i] /= peak
        return chroma
    }

    private fun detectBassPitchClass(mags: DoubleArray, n: Int): Int {
        var bassMax = 0.0
        var firstBin = -1
        var lastBin = -1
        for (bin in 1 until mags.size - 1) {
            val freq = bin * sampleRate.toDouble() / n
            if (freq < profile.minFreq) continue
            if (freq > profile.bassCeiling) break
            if (firstBin < 0) firstBin = bin
            lastBin = bin
            if (mags[bin] > bassMax) bassMax = mags[bin]
        }
        if (firstBin < 0 || bassMax <= 0.0) return -1
        val threshold = bassMax * 0.30
        for (bin in firstBin..lastBin) {
            if (mags[bin] < threshold) continue
            if (bin > 0 && bin < mags.lastIndex &&
                mags[bin] >= mags[bin - 1] && mags[bin] >= mags[bin + 1]) {
                val freq = (bin + FFT.interpolatePeak(mags, bin)) * sampleRate.toDouble() / n
                val midi = 69 + 12 * (ln(freq / 440.0) / ln(2.0))
                return ((midi.roundToInt() % 12) + 12) % 12
            }
        }
        return -1
    }

    private fun scoreTemplates(c: DoubleArray, bass: Int, minScore: Double): ChordResult? {
        var best: Template? = null
        var bestScore = Double.NEGATIVE_INFINITY
        var secondScore = Double.NEGATIVE_INFINITY

        for (template in templates) {
            var inChord = 0.0
            var outChord = 0.0
            var covered = 0
            for (pc in 0 until 12) {
                if (template.mask[pc]) {
                    inChord += c[pc]
                    if (c[pc] >= 0.22) covered++
                } else {
                    outChord += c[pc]
                }
            }
            val requiredTones = if (template.pitches.size <= 3) 2 else 3
            if (covered < requiredTones) continue

            val inMean = inChord / template.pitches.size
            val outMean = outChord / (12 - template.pitches.size)
            val coverage = covered.toDouble() / template.pitches.size
            val specificityBonus = 0.05 * (template.pitches.size - 3) * coverage
            var score = 0.70 * inMean + 0.22 * coverage - 0.38 * outMean + specificityBonus
            if (bass == template.rootPc) score += 0.08

            if (score > bestScore) {
                secondScore = bestScore
                bestScore = score
                best = template
            } else if (score > secondScore) {
                secondScore = score
            }
        }

        val winner = best ?: return null
        if (bestScore < minScore) return null
        val confidence = if (secondScore.isFinite()) (bestScore - secondScore).coerceAtLeast(0.0) else bestScore
        return ChordResult(winner.name, bestScore, confidence, bass)
    }
}
