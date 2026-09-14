package com.nicos.pitchkit.tuner.harmony.benchmark

import com.nicos.pitchkit.tuner.harmony.song.SongChordSegment
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HarmonyBenchmarkTest {
    @Test
    fun scoresComplexChordByLabelRootBassAndPitches() {
        val expected = listOf(
            HarmonyBenchmarkCase(
                id = "g-six-nine",
                startMs = 1_000,
                endMs = 2_000,
                acceptedLabels = setOf("G6/9"),
                expectedRoot = "G",
                expectedBass = "G",
                expectedPitchClasses = setOf("G", "B", "D", "E", "A"),
            )
        )
        val detected = listOf(
            SongChordSegment(
                label = "G6/9",
                startMs = 900,
                endMs = 2_100,
                confidence = 0.8,
                root = "G",
                bass = "G",
                pitchClasses = listOf("G", "B", "D", "E", "A"),
            )
        )

        val result = HarmonyBenchmarkEvaluator.evaluate(expected, detected)
        assertEquals(1.0, result.coverage, 1e-9)
        assertEquals(1.0, result.labelAccuracy ?: 0.0, 1e-9)
        assertEquals(1.0, result.rootAccuracy ?: 0.0, 1e-9)
        assertEquals(1.0, result.pitchF1 ?: 0.0, 1e-9)
    }

    @Test
    fun acceptsHalfDiminishedAliasAndEnharmonicRoot() {
        val expected = listOf(
            HarmonyBenchmarkCase(
                id = "half-dim",
                startMs = 100,
                acceptedLabels = setOf("F#m7b5"),
                expectedRoot = "Gb",
            )
        )
        val detected = listOf(
            SongChordSegment(
                label = "F#ø7",
                startMs = 0,
                endMs = 500,
                confidence = 0.7,
            )
        )

        val result = HarmonyBenchmarkEvaluator.evaluate(expected, detected)
        assertTrue(result.cases.single().labelCorrect == true)
        assertTrue(result.cases.single().rootCorrect == true)
    }
}
