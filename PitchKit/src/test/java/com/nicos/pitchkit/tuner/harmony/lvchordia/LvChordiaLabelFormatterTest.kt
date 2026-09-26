package com.nicos.pitchkit.tuner.harmony.lvchordia

import org.junit.Assert.assertEquals
import org.junit.Test

class LvChordiaLabelFormatterTest {
    @Test
    fun formatsHalfDiminishedAndSixNineClearly() {
        val formatter = LvChordiaLabelFormatter(preferFlats = false)

        assertEquals("F#ø7", formatter.format("F#:hdim7"))
        assertEquals("G6/9", formatter.format("G:maj6(9)"))
        assertEquals("G6/9", formatter.format("G:maj(6,9)"))
        assertEquals("G6/9/B", formatter.format("G:maj6(9)/3"))
    }

    @Test
    fun `harte suspensions never borrow a third`() {
        val formatter = LvChordiaLabelFormatter(preferFlats = false)

        assertEquals("Dsus4", formatter.format("D:sus4"))
        assertEquals("D7sus4", formatter.format("D:sus4(b7)"))
        assertEquals("D9sus4", formatter.format("D:sus4(b7,9)"))
        assertEquals("D13sus4", formatter.format("D:sus4(b7,9,13)"))
        assertEquals("Dsus2", formatter.format("D:sus2"))
        assertEquals("Dsus2(4)", formatter.format("D:sus2(4)"))
        assertEquals("Dsus2(b7)", formatter.format("D:sus2(b7)"))
        assertEquals("D7sus4/E", formatter.format("D:sus4(b7)/2"))
    }

    @Test
    fun `unhandled extensions keep their degree list after the mapped quality`() {
        val formatter = LvChordiaLabelFormatter(preferFlats = false)

        assertEquals("Dm7(4)", formatter.format("D:min7(4)"))
        assertEquals("Dm(11)", formatter.format("D:min(11)"))
        assertEquals("D(9)", formatter.format("D:maj(9)"))
        assertEquals("D6(#11)", formatter.format("D:maj6(#11)"))
        assertEquals("Daug(b7)", formatter.format("D:aug(b7)"))
    }
}
