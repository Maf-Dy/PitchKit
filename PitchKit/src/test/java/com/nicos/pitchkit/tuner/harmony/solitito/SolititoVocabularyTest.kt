package com.nicos.pitchkit.tuner.harmony.solitito

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The eleven quality classes onto the labels the live screen renders, and onto the
 * interval sets the benchmark scores.
 *
 * Every one of these is a claim about what a player sees, and two of them are traps:
 *
 *  * **`dim7`, not `dim`.** `brain.rs` spells the *fully diminished seventh* `dim`.
 *    Reading it as a diminished triad would silently drop the diminished seventh from
 *    every extension metric and show the player a chord one note short.
 *  * **`m7b5` is Harte's `hdim7`.** Same four notes, and the app's own parser reads both
 *    spellings, but the label solitito emits has to be one the parser knows.
 *
 * The `intervals` column is `live_metrics.SUFFIX_INTERVALS`, which is what the live
 * benchmark scores these labels against; it is pinned here so a relabelling cannot
 * quietly change what "correct" means.
 */
class SolititoVocabularyTest {

    /** quality -> display label for root C -> the pitch classes that label means. */
    private val expected = listOf(
        Triple("maj", "C", setOf(0, 4, 7)),
        Triple("min", "Cm", setOf(0, 3, 7)),
        Triple("maj7", "Cmaj7", setOf(0, 4, 7, 11)),
        Triple("dom7", "C7", setOf(0, 4, 7, 10)),
        Triple("min7", "Cm7", setOf(0, 3, 7, 10)),
        Triple("m7b5", "Cm7b5", setOf(0, 3, 6, 10)),
        Triple("dim7", "Cdim7", setOf(0, 3, 6, 9)),
        Triple("aug", "Caug", setOf(0, 4, 8)),
        Triple("sus", "Csus4", setOf(0, 5, 7)),
    )

    @Test
    fun theElevenClassesAreTheOnesTheModelHas() {
        assertEquals(SolititoContract.QUALITY_CLASSES, SolititoVocabulary.QUALITIES.size)
        assertEquals(SolititoContract.ROOT_CLASSES, SolititoVocabulary.ROOTS.size)
        assertEquals("Noise", SolititoVocabulary.ROOTS.last())
        assertEquals(
            listOf("maj", "min", "maj7", "dom7", "min7", "m7b5", "dim7", "aug", "sus",
                "note", "N"),
            SolititoVocabulary.QUALITIES.toList(),
        )
        // Nine chord qualities and two meta classes; the head order is the trained one.
        assertEquals(9, expected.size)
        assertEquals(expected.map { it.first }, SolititoVocabulary.QUALITIES.dropLast(2))
    }

    @Test
    fun everyChordQualityRoundTripsToItsDisplayLabel() {
        for ((quality, label, _) in expected) {
            val chord = SolititoVocabulary.chordString(0, quality)
            assertEquals("$quality display", label, SolititoVocabulary.toDisplayLabel(chord))
            assertTrue("$quality must be a real chord", SolititoVocabulary.isRealChord(chord))
            assertEquals("$quality root", "C", SolititoVocabulary.rootOf(chord))
        }
        // ...and on every root, with the sharp spellings the model's classes carry.
        assertEquals("G#m7b5", SolititoVocabulary.toDisplayLabel(
            SolititoVocabulary.chordString(8, "m7b5")))
        assertEquals("A#dim7", SolititoVocabulary.toDisplayLabel(
            SolititoVocabulary.chordString(10, "dim7")))
        assertEquals("B", SolititoVocabulary.toDisplayLabel(
            SolititoVocabulary.chordString(11, "maj")))

        // The nine labels name nine different pitch sets, and the one pair that is easy
        // to conflate differs by exactly its seventh: dim7 has a 9, m7b5 has a 10.
        val intervals = expected.associate { it.second to it.third }
        assertEquals(9, intervals.values.distinct().size)
        val dim7 = intervals.getValue("Cdim7")
        val halfDim = intervals.getValue("Cm7b5")
        assertEquals(setOf(0, 3, 6), dim7.intersect(halfDim))
        assertEquals(setOf(9), dim7.minus(halfDim))
        assertEquals(setOf(10), halfDim.minus(dim7))
    }

    @Test
    fun theThreeRefusalsAllShowNothing() {
        // A single note, the `N` quality class, and the `Noise` root.
        assertEquals("Note C", SolititoVocabulary.chordString(0, "note"))
        assertEquals("N", SolititoVocabulary.toDisplayLabel(SolititoVocabulary.chordString(0, "note")))
        assertEquals("Noise", SolititoVocabulary.chordString(0, "N"))
        assertEquals("Noise", SolititoVocabulary.chordString(12, "maj"))
        assertEquals("N", SolititoVocabulary.toDisplayLabel("Noise"))
        // ...and the window that was asked about but is not live enough to be named.
        assertEquals("N", SolititoVocabulary.toDisplayLabel(SolititoVocabulary.PENDING))
        assertEquals("N", SolititoVocabulary.toDisplayLabel(""))

        for (chord in listOf("Noise", SolititoVocabulary.PENDING, "Note C", "")) {
            assertFalse("$chord must not be lockable", SolititoVocabulary.isRealChord(chord))
        }
    }

    @Test
    fun onlyTheFourNoteQualitiesDecay() {
        // A seventh losing a note is a decay of itself; a triad has nothing to lose.
        assertTrue(SolititoVocabulary.isDecayOf("C m7", "C m"))
        assertTrue(SolititoVocabulary.isDecayOf("C Maj7", "C"))
        assertTrue(SolititoVocabulary.isDecayOf("C 7", "C sus4"))
        assertTrue(SolititoVocabulary.isDecayOf("C dim", "C"))
        assertTrue(SolititoVocabulary.isDecayOf("C m7b5", "Note C"))
        // A four-note chord becoming another four-note chord is a correction, not a decay.
        assertFalse(SolititoVocabulary.isDecayOf("C m7", "C m7b5"))
        // A triad is not a decay of anything.
        assertFalse(SolititoVocabulary.isDecayOf("C", "C m"))
        // Neither is a different root.
        assertFalse(SolititoVocabulary.isDecayOf("C m7", "G m"))
    }

    @Test
    fun theVocabularyHasNoSixthsNinthsInversionsOrBass() {
        // Stated as a test because it is the model's ceiling and the reason its extension
        // precision never over-claims: a C6 can only come back as its A m7, and a C m6 as
        // its A m7b5 - a different root and a different family, but not one wrong note.
        val labels = expected.map { it.second }
        assertTrue(labels.none { it.contains("6") || it.contains("9") || it.contains("/") })
        assertEquals(9, labels.distinct().size)
    }
}
