package com.nicos.pitchkit.tuner.harmony

import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

internal data class PitchClassRerankResult(
    val label: String,
    val changed: Boolean,
    val originalScore: Double,
    val score: Double,
)

/**
 * Conservative second opinion for chord spelling.
 *
 * The neural recognizer remains responsible for the broad harmonic family. This
 * helper resolves closely-related spellings from direct pitch-class evidence
 * (7 vs 6/9, dim7 vs hdim7, etc.) and one exact pitch-set ambiguity: m6 vs hdim7,
 * where low-register evidence can identify the intended root.
 */
internal object PitchClassChordReranker {
    private data class Quality(
        val suffix: String,
        val family: Family,
        val intervals: IntArray,
        val triadIntervals: IntArray,
    )

    private enum class Family { MAJOR, MINOR, DIMINISHED, SUSPENDED, AUGMENTED }

    private val roots = mapOf(
        "C" to 0, "C#" to 1, "Db" to 1, "D" to 2, "D#" to 3, "Eb" to 3,
        "E" to 4, "F" to 5, "F#" to 6, "Gb" to 6, "G" to 7,
        "G#" to 8, "Ab" to 8, "A" to 9, "A#" to 10, "Bb" to 10, "B" to 11,
    )
    private val sharpNames = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    private val flatNames = arrayOf("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

    private val qualities = listOf(
        Quality("", Family.MAJOR, intArrayOf(0, 4, 7), intArrayOf(0, 4, 7)),
        Quality("6", Family.MAJOR, intArrayOf(0, 4, 7, 9), intArrayOf(0, 4, 7)),
        Quality("6/9", Family.MAJOR, intArrayOf(0, 2, 4, 7, 9), intArrayOf(0, 4, 7)),
        Quality("7", Family.MAJOR, intArrayOf(0, 4, 7, 10), intArrayOf(0, 4, 7)),
        Quality("maj7", Family.MAJOR, intArrayOf(0, 4, 7, 11), intArrayOf(0, 4, 7)),
        Quality("9", Family.MAJOR, intArrayOf(0, 2, 4, 7, 10), intArrayOf(0, 4, 7)),
        Quality("maj9", Family.MAJOR, intArrayOf(0, 2, 4, 7, 11), intArrayOf(0, 4, 7)),
        Quality("m", Family.MINOR, intArrayOf(0, 3, 7), intArrayOf(0, 3, 7)),
        Quality("m6", Family.MINOR, intArrayOf(0, 3, 7, 9), intArrayOf(0, 3, 7)),
        Quality("m7", Family.MINOR, intArrayOf(0, 3, 7, 10), intArrayOf(0, 3, 7)),
        Quality("m(maj7)", Family.MINOR, intArrayOf(0, 3, 7, 11), intArrayOf(0, 3, 7)),
        Quality("m9", Family.MINOR, intArrayOf(0, 2, 3, 7, 10), intArrayOf(0, 3, 7)),
        Quality("dim", Family.DIMINISHED, intArrayOf(0, 3, 6), intArrayOf(0, 3, 6)),
        Quality("dim7", Family.DIMINISHED, intArrayOf(0, 3, 6, 9), intArrayOf(0, 3, 6)),
        Quality("ø7", Family.DIMINISHED, intArrayOf(0, 3, 6, 10), intArrayOf(0, 3, 6)),
        Quality("sus2", Family.SUSPENDED, intArrayOf(0, 2, 7), intArrayOf(0, 2, 7)),
        Quality("sus4", Family.SUSPENDED, intArrayOf(0, 5, 7), intArrayOf(0, 5, 7)),
        Quality("aug", Family.AUGMENTED, intArrayOf(0, 4, 8), intArrayOf(0, 4, 8)),
    )

    fun rerank(label: String, evidence: FloatArray): PitchClassRerankResult {
        require(evidence.size == 12)
        val parsed = parse(label) ?: return unchanged(label)
        val normalized = normalize(evidence)
        val original = qualities.firstOrNull { it.suffix == parsed.suffix }
            ?: return unchanged(label)

        val allowedFamilies = mutableSetOf(original.family)
        val diminishedFifth = normalized[(parsed.rootPc + 6) % 12]
        val perfectFifth = normalized[(parsed.rootPc + 7) % 12]
        if (original.family == Family.MINOR && diminishedFifth > perfectFifth + 0.14f) {
            allowedFamilies += Family.DIMINISHED
        } else if (original.family == Family.DIMINISHED && perfectFifth > diminishedFifth + 0.14f) {
            allowedFamilies += Family.MINOR
        }

        val candidates = qualities.filter { it.family in allowedFamilies }
        val originalScore = score(original, parsed.rootPc, normalized)
        val best = candidates.maxByOrNull { score(it, parsed.rootPc, normalized) } ?: original
        val bestScore = score(best, parsed.rootPc, normalized)

        // Do not rewrite a model decision on weak/marginal evidence. Requiring a
        // real score margin is what keeps passing melody tones from becoming
        // invented chord extensions.
        val changed = best.suffix != original.suffix &&
            bestScore >= originalScore + 0.055 &&
            requiredTonesSupported(best, parsed.rootPc, normalized)

        val chosen = if (changed) best else original
        val rendered = buildString {
            append(parsed.rootText)
            append(chosen.suffix)
            parsed.inversion?.let {
                append('/')
                append(it)
            }
        }
        return PitchClassRerankResult(
            label = rendered,
            changed = changed,
            originalScore = originalScore,
            score = if (changed) bestScore else originalScore,
        )
    }

    /**
     * Resolve the exact pitch-set equivalence Rm6 == (R-3)ø7 from low-register
     * evidence. This is intentionally the only cross-root rewrite supported.
     */
    fun resolveEquivalentRoot(
        label: String,
        pitchEvidence: FloatArray,
        bassEvidence: FloatArray,
    ): PitchClassRerankResult {
        require(pitchEvidence.size == 12)
        require(bassEvidence.size == 12)
        val parsed = parse(label) ?: return unchanged(label)
        if (parsed.inversion != null) return unchanged(label)
        if (parsed.suffix != "m6" && parsed.suffix != "ø7") return unchanged(label)

        val targetRoot = if (parsed.suffix == "m6") {
            floorMod12(parsed.rootPc - 3)
        } else {
            floorMod12(parsed.rootPc + 3)
        }
        val targetSuffix = if (parsed.suffix == "m6") "ø7" else "m6"
        val targetQuality = qualities.first { it.suffix == targetSuffix }
        val normalizedPitch = normalize(pitchEvidence)
        val normalizedBass = normalize(bassEvidence)

        if (!requiredTonesSupported(targetQuality, targetRoot, normalizedPitch)) {
            return unchanged(label)
        }

        val currentBass = normalizedBass[parsed.rootPc].toDouble()
        val targetBass = normalizedBass[targetRoot].toDouble()
        val switch = targetBass >= 0.32 && targetBass >= currentBass + 0.12
        if (!switch) return unchanged(label)

        val useFlats = parsed.rootText.contains('b')
        val targetRootText = if (useFlats) flatNames[targetRoot] else sharpNames[targetRoot]
        return PitchClassRerankResult(
            label = targetRootText + targetSuffix,
            changed = true,
            originalScore = currentBass,
            score = targetBass,
        )
    }

    /** Fold CQT bins into 12 pitch classes over the newest frames. */
    fun cqtEvidence(
        values: FloatArray,
        frameCount: Int,
        binCount: Int,
        fmin: Double,
        binsPerOctave: Int,
        logMagnitude: Boolean,
        tailFrames: Int = 4,
    ): FloatArray = foldCqt(
        values = values,
        frameCount = frameCount,
        binCount = binCount,
        fmin = fmin,
        binsPerOctave = binsPerOctave,
        logMagnitude = logMagnitude,
        tailFrames = tailFrames,
        maxFrequencyHz = Double.POSITIVE_INFINITY,
        bassWeighting = false,
    )

    /** Low-register-only CQT evidence used to disambiguate m6 from hdim7 roots. */
    fun cqtBassEvidence(
        values: FloatArray,
        frameCount: Int,
        binCount: Int,
        fmin: Double,
        binsPerOctave: Int,
        logMagnitude: Boolean,
        tailFrames: Int = 4,
        maxFrequencyHz: Double = 420.0,
    ): FloatArray = foldCqt(
        values = values,
        frameCount = frameCount,
        binCount = binCount,
        fmin = fmin,
        binsPerOctave = binsPerOctave,
        logMagnitude = logMagnitude,
        tailFrames = tailFrames,
        maxFrequencyHz = maxFrequencyHz,
        bassWeighting = true,
    )

    private fun foldCqt(
        values: FloatArray,
        frameCount: Int,
        binCount: Int,
        fmin: Double,
        binsPerOctave: Int,
        logMagnitude: Boolean,
        tailFrames: Int,
        maxFrequencyHz: Double,
        bassWeighting: Boolean,
    ): FloatArray {
        if (frameCount <= 0 || binCount <= 0 || values.size < frameCount * binCount) {
            return FloatArray(12)
        }
        val result = DoubleArray(12)
        val firstFrame = (frameCount - tailFrames.coerceAtLeast(1)).coerceAtLeast(0)
        for (frame in firstFrame until frameCount) {
            val offset = frame * binCount
            for (bin in 0 until binCount) {
                val frequency = fmin * 2.0.pow(bin.toDouble() / binsPerOctave.toDouble())
                if (frequency > maxFrequencyHz) continue
                val midi = (69.0 + 12.0 * log2(frequency / 440.0)).roundToInt()
                val pitchClass = floorMod12(midi)
                val raw = values[offset + bin].toDouble()
                val magnitude = if (logMagnitude) {
                    kotlin.math.exp(raw).coerceAtMost(1e12)
                } else {
                    raw.coerceAtLeast(0.0)
                }
                val compressed = sqrt(magnitude.coerceAtLeast(0.0))
                val weight = if (bassWeighting) {
                    sqrt((fmin / frequency).coerceAtMost(1.0))
                } else {
                    1.0
                }
                result[pitchClass] += compressed * weight
            }
        }
        return normalize(result)
    }

    /** Average already-normalized per-frame chroma rows. */
    fun averageEvidence(
        chroma: FloatArray,
        frameCount: Int,
        frameIndices: IntRange,
    ): FloatArray {
        if (frameCount <= 0 || chroma.size < frameCount * 12) return FloatArray(12)
        val result = DoubleArray(12)
        var count = 0
        val first = frameIndices.first.coerceIn(0, frameCount - 1)
        val last = frameIndices.last.coerceIn(first, frameCount - 1)
        for (frame in first..last) {
            val offset = frame * 12
            for (pc in 0 until 12) result[pc] += chroma[offset + pc]
            count++
        }
        if (count > 0) for (pc in 0 until 12) result[pc] /= count.toDouble()
        return normalize(result)
    }

    private fun score(quality: Quality, root: Int, evidence: FloatArray): Double {
        val required = quality.intervals.map { evidence[(root + it) % 12].toDouble() }
        val mean = required.average()
        val weakest = required.minOrNull() ?: 0.0
        val requiredSet = quality.intervals.map { (root + it) % 12 }.toSet()
        val strongestOutside = evidence.indices
            .filter { it !in requiredSet }
            .maxOfOrNull { evidence[it].toDouble() }
            ?: 0.0

        var value = 0.56 * mean + 0.34 * weakest - 0.18 * strongestOutside
        // Added tones must earn their way into the spelling.
        val extensionCount = (quality.intervals.size - quality.triadIntervals.size).coerceAtLeast(0)
        value -= 0.012 * extensionCount
        return value
    }

    private fun requiredTonesSupported(quality: Quality, root: Int, evidence: FloatArray): Boolean {
        val triad = quality.triadIntervals.map { evidence[(root + it) % 12] }
        if (triad.any { it < 0.16f }) return false
        val triadSet = quality.triadIntervals.toSet()
        val extensionIntervals = quality.intervals.filter { it !in triadSet }
        if (extensionIntervals.isEmpty()) return true
        val triadMean = triad.average().toFloat()
        val threshold = maxOf(0.18f, triadMean * 0.28f)
        return extensionIntervals.all { evidence[(root + it) % 12] >= threshold }
    }

    private data class Parsed(
        val rootText: String,
        val rootPc: Int,
        val suffix: String,
        val inversion: String?,
    )

    private fun parse(label: String): Parsed? {
        if (label.isBlank()) return null

        val rootText = when {
            label.length >= 2 && (label[1] == '#' || label[1] == 'b') -> label.substring(0, 2)
            else -> label.substring(0, 1)
        }
        val rootPc = roots[rootText] ?: return null
        val remainder = label.substring(rootText.length)

        // 6/9 contains a slash as part of the quality, not as an inversion marker.
        val suffix: String
        val inversion: String?
        if (remainder.startsWith("6/9")) {
            suffix = "6/9"
            inversion = if (remainder.startsWith("6/9/") && remainder.length > 4) {
                remainder.substring(4)
            } else {
                null
            }
        } else {
            val slash = remainder.indexOf('/')
            suffix = if (slash >= 0) remainder.substring(0, slash) else remainder
            inversion = if (slash >= 0 && slash + 1 < remainder.length) {
                remainder.substring(slash + 1)
            } else {
                null
            }
        }
        return Parsed(rootText, rootPc, suffix, inversion)
    }

    private fun normalize(values: FloatArray): FloatArray = normalize(DoubleArray(12) { values[it].toDouble() })

    private fun normalize(values: DoubleArray): FloatArray {
        val peak = values.maxOrNull()?.coerceAtLeast(0.0) ?: 0.0
        if (peak <= 1e-12) return FloatArray(12)
        return FloatArray(12) { index -> (values[index] / peak).coerceIn(0.0, 1.0).toFloat() }
    }

    private fun unchanged(label: String) = PitchClassRerankResult(label, false, 0.0, 0.0)

    private fun log2(value: Double): Double = ln(value) / ln(2.0)
    private fun floorMod12(value: Int): Int = ((value % 12) + 12) % 12
}
