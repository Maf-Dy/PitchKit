package com.nicos.pitchkit.tuner.harmony.song

import kotlin.math.max
import kotlin.math.min

/** Coarse repeated harmonic A/B/C section grouping shared by offline analyzers. */
object SongSectionDetector {
    private const val SECTION_TARGET_CHORDS = 8
    private const val MIN_SECTION_MS = 10_000L
    private const val MAX_SECTION_MS = 20_000L
    private const val SECTION_MATCH_THRESHOLD = 0.62

    fun detect(
        chords: List<SongChordSegment>,
        durationMs: Long,
    ): List<SongSection> {
        if (durationMs <= 0L) return emptyList()
        if (chords.isEmpty()) return listOf(SongSection("A", 0L, durationMs))

        val durations = chords.map { max(1L, it.endMs - it.startMs) }.sorted()
        val medianChordMs = durations[durations.size / 2]
        val blockMs = (medianChordMs * SECTION_TARGET_CHORDS)
            .coerceIn(MIN_SECTION_MS, MAX_SECTION_MS)

        data class Template(val label: String, val signature: List<String>)
        val templates = mutableListOf<Template>()
        val blocks = mutableListOf<SongSection>()

        var start = 0L
        while (start < durationMs) {
            val end = min(durationMs, start + blockMs)
            val signature = chordSignature(chords, start, end)
            var bestTemplate: Template? = null
            var bestSimilarity = 0.0
            for (template in templates) {
                val similarity = signatureSimilarity(signature, template.signature)
                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity
                    bestTemplate = template
                }
            }

            val label = if (bestTemplate != null && bestSimilarity >= SECTION_MATCH_THRESHOLD) {
                bestTemplate.label
            } else {
                sectionLabel(templates.size).also {
                    templates += Template(it, signature)
                }
            }
            blocks += SongSection(label, start, end)
            start = end
        }

        val merged = mutableListOf<SongSection>()
        for (block in blocks) {
            val previous = merged.lastOrNull()
            if (previous != null && previous.label == block.label) {
                merged[merged.lastIndex] = previous.copy(endMs = block.endMs)
            } else {
                merged += block
            }
        }
        return merged
    }

    private fun chordSignature(
        chords: List<SongChordSegment>,
        startMs: Long,
        endMs: Long,
    ): List<String> {
        val result = mutableListOf<String>()
        for (chord in chords) {
            if (chord.endMs <= startMs || chord.startMs >= endMs) continue
            if (result.lastOrNull() != chord.label) result += chord.label
        }
        return result.take(12)
    }

    private fun signatureSimilarity(first: List<String>, second: List<String>): Double {
        if (first.isEmpty() && second.isEmpty()) return 1.0
        if (first.isEmpty() || second.isEmpty()) return 0.0
        val lcs = Array(first.size + 1) { IntArray(second.size + 1) }
        for (i in 1..first.size) {
            for (j in 1..second.size) {
                lcs[i][j] = if (first[i - 1] == second[j - 1]) {
                    lcs[i - 1][j - 1] + 1
                } else {
                    max(lcs[i - 1][j], lcs[i][j - 1])
                }
            }
        }
        return 2.0 * lcs[first.size][second.size] / (first.size + second.size).toDouble()
    }

    private fun sectionLabel(index: Int): String {
        val letter = ('A'.code + (index % 26)).toChar()
        val cycle = index / 26
        return if (cycle == 0) letter.toString() else "$letter${cycle + 1}"
    }
}
