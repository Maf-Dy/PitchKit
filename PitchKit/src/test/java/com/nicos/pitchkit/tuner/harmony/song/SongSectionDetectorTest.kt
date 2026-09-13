package com.nicos.pitchkit.tuner.harmony.song

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SongSectionDetectorTest {
    @Test
    fun repeatedProgressionReusesSectionLabel() {
        val progression = listOf("C", "G", "Am", "F")
        val chords = buildList {
            var start = 0L
            repeat(4) {
                progression.forEach { chord ->
                    add(SongChordSegment(chord, start, start + 2_000L, 0.9))
                    start += 2_000L
                }
            }
        }

        val sections = SongSectionDetector.detect(chords, 32_000L)

        assertTrue(sections.isNotEmpty())
        assertEquals("A", sections.first().label)
        assertTrue(sections.all { it.label == "A" })
    }

    @Test
    fun emptyHarmonyStillProducesOneSectionForKnownDuration() {
        assertEquals(
            listOf(SongSection("A", 0L, 12_000L)),
            SongSectionDetector.detect(emptyList(), 12_000L),
        )
    }
}
