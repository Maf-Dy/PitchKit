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
}
