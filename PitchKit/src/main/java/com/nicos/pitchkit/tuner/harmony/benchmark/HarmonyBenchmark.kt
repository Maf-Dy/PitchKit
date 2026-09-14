package com.nicos.pitchkit.tuner.harmony.benchmark

import com.nicos.pitchkit.tuner.harmony.song.SongChordSegment
import kotlin.math.max
import kotlin.math.min

/**
 * Ground-truth point/interval used to compare offline engines on the same music.
 * Multiple accepted labels are allowed because complex harmony can have more
 * than one defensible chord spelling; root/bass/pitch scoring remains separate.
 */
data class HarmonyBenchmarkCase(
    val id: String,
    val startMs: Long,
    val endMs: Long = startMs,
    val acceptedLabels: Set<String> = emptySet(),
    val expectedRoot: String? = null,
    val expectedBass: String? = null,
    val expectedPitchClasses: Set<String> = emptySet(),
) {
    init {
        require(id.isNotBlank())
        require(startMs >= 0L)
        require(endMs >= startMs)
        require(
            acceptedLabels.isNotEmpty() ||
                expectedRoot != null ||
                expectedBass != null ||
                expectedPitchClasses.isNotEmpty()
        ) { "Benchmark case '$id' has no expected harmony information" }
    }
}

data class HarmonyBenchmarkCaseResult(
    val case: HarmonyBenchmarkCase,
    val detected: SongChordSegment?,
    val labelCorrect: Boolean?,
    val rootCorrect: Boolean?,
    val bassCorrect: Boolean?,
    val pitchPrecision: Double?,
    val pitchRecall: Double?,
    val pitchF1: Double?,
)

data class HarmonyBenchmarkSummary(
    val caseCount: Int,
    val detectedCount: Int,
    val coverage: Double,
    val labelAccuracy: Double?,
    val rootAccuracy: Double?,
    val bassAccuracy: Double?,
    val pitchPrecision: Double?,
    val pitchRecall: Double?,
    val pitchF1: Double?,
    val cases: List<HarmonyBenchmarkCaseResult>,
)

object HarmonyBenchmarkEvaluator {
    fun evaluate(
        expected: List<HarmonyBenchmarkCase>,
        detected: List<SongChordSegment>,
    ): HarmonyBenchmarkSummary {
        val results = expected.map { benchmarkCase ->
            val match = bestOverlappingSegment(benchmarkCase, detected)
            evaluateCase(benchmarkCase, match)
        }

        return HarmonyBenchmarkSummary(
            caseCount = results.size,
            detectedCount = results.count { it.detected != null },
            coverage = ratio(results.count { it.detected != null }, results.size),
            labelAccuracy = nullableAccuracy(results.mapNotNull { it.labelCorrect }),
            rootAccuracy = nullableAccuracy(results.mapNotNull { it.rootCorrect }),
            bassAccuracy = nullableAccuracy(results.mapNotNull { it.bassCorrect }),
            pitchPrecision = averageOrNull(results.mapNotNull { it.pitchPrecision }),
            pitchRecall = averageOrNull(results.mapNotNull { it.pitchRecall }),
            pitchF1 = averageOrNull(results.mapNotNull { it.pitchF1 }),
            cases = results,
        )
    }

    private fun evaluateCase(
        benchmarkCase: HarmonyBenchmarkCase,
        detected: SongChordSegment?,
    ): HarmonyBenchmarkCaseResult {
        val acceptedLabels = benchmarkCase.acceptedLabels.map(::normalizeLabel).toSet()
        val labelCorrect = if (acceptedLabels.isEmpty()) {
            null
        } else {
            detected?.let { normalizeLabel(it.label) in acceptedLabels } ?: false
        }

        val expectedRootPc = benchmarkCase.expectedRoot?.let(::pitchClass)
        val actualRootPc = detected?.root?.let(::pitchClass)
            ?: detected?.label?.let(::rootFromLabel)?.let(::pitchClass)
        val rootCorrect = expectedRootPc?.let { expected -> actualRootPc == expected }

        val expectedBassPc = benchmarkCase.expectedBass?.let(::pitchClass)
        val actualBassPc = detected?.bass?.let(::pitchClass)
            ?: detected?.label?.let(::bassFromLabel)?.let(::pitchClass)
        val bassCorrect = expectedBassPc?.let { expected -> actualBassPc == expected }

        val expectedPitches = benchmarkCase.expectedPitchClasses.mapNotNull(::pitchClass).toSet()
        val actualPitches = detected?.pitchClasses.orEmpty().mapNotNull(::pitchClass).toSet()
        val (precision, recall, f1) = if (expectedPitches.isEmpty()) {
            Triple(null, null, null)
        } else if (actualPitches.isEmpty()) {
            Triple(0.0, 0.0, 0.0)
        } else {
            val truePositive = expectedPitches.intersect(actualPitches).size
            val p = ratio(truePositive, actualPitches.size)
            val r = ratio(truePositive, expectedPitches.size)
            val harmonic = if (p + r > 0.0) 2.0 * p * r / (p + r) else 0.0
            Triple(p, r, harmonic)
        }

        return HarmonyBenchmarkCaseResult(
            case = benchmarkCase,
            detected = detected,
            labelCorrect = labelCorrect,
            rootCorrect = rootCorrect,
            bassCorrect = bassCorrect,
            pitchPrecision = precision,
            pitchRecall = recall,
            pitchF1 = f1,
        )
    }

    private fun bestOverlappingSegment(
        benchmarkCase: HarmonyBenchmarkCase,
        detected: List<SongChordSegment>,
    ): SongChordSegment? {
        if (detected.isEmpty()) return null
        val queryEnd = if (benchmarkCase.endMs > benchmarkCase.startMs) {
            benchmarkCase.endMs
        } else {
            benchmarkCase.startMs + 1L
        }

        return detected
            .asSequence()
            .map { segment ->
                val overlap = overlapMs(
                    benchmarkCase.startMs,
                    queryEnd,
                    segment.startMs,
                    segment.endMs,
                )
                segment to overlap
            }
            .filter { it.second > 0L }
            .maxWithOrNull(
                compareBy<Pair<SongChordSegment, Long>> { it.second }
                    .thenBy { it.first.confidence }
            )
            ?.first
    }

    private fun overlapMs(aStart: Long, aEnd: Long, bStart: Long, bEnd: Long): Long =
        max(0L, min(aEnd, bEnd) - max(aStart, bStart))

    private fun normalizeLabel(label: String): String = label
        .trim()
        .replace("m7b5", "ø7", ignoreCase = true)
        .replace("half-dim", "ø7", ignoreCase = true)
        .replace("♭", "b")
        .replace("♯", "#")
        .replace(" ", "")
        .lowercase()

    private fun rootFromLabel(label: String): String? {
        val trimmed = label.trim()
        if (trimmed.isEmpty() || trimmed[0].uppercaseChar() !in 'A'..'G') return null
        return if (trimmed.length >= 2 && (trimmed[1] == '#' || trimmed[1] == 'b' || trimmed[1] == '♯' || trimmed[1] == '♭')) {
            trimmed.substring(0, 2)
        } else {
            trimmed.substring(0, 1)
        }
    }

    private fun bassFromLabel(label: String): String? {
        // 6/9 is a quality token, not an inversion. An actual inversion follows
        // another slash, e.g. G6/9/B.
        val root = rootFromLabel(label) ?: return null
        val remainder = label.substring(root.length)
        val slashIndex = if (remainder.startsWith("6/9")) {
            remainder.indexOf('/', startIndex = 3)
        } else {
            remainder.lastIndexOf('/')
        }
        if (slashIndex < 0 || slashIndex + 1 >= remainder.length) return root
        return rootFromLabel(remainder.substring(slashIndex + 1)) ?: root
    }

    private fun pitchClass(note: String): Int? {
        val normalized = note.trim()
            .replace("♭", "b")
            .replace("♯", "#")
        if (normalized.isEmpty()) return null
        return when (normalized.lowercase()) {
            "c", "b#" -> 0
            "c#", "db" -> 1
            "d" -> 2
            "d#", "eb" -> 3
            "e", "fb" -> 4
            "f", "e#" -> 5
            "f#", "gb" -> 6
            "g" -> 7
            "g#", "ab" -> 8
            "a" -> 9
            "a#", "bb" -> 10
            "b", "cb" -> 11
            else -> null
        }
    }

    private fun nullableAccuracy(values: List<Boolean>): Double? =
        if (values.isEmpty()) null else ratio(values.count { it }, values.size)

    private fun averageOrNull(values: List<Double>): Double? =
        if (values.isEmpty()) null else values.average()

    private fun ratio(numerator: Int, denominator: Int): Double =
        if (denominator <= 0) 0.0 else numerator.toDouble() / denominator.toDouble()
}
