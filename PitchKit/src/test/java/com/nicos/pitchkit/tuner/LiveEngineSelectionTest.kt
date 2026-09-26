package com.nicos.pitchkit.tuner

import android.media.MediaRecorder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The shipped live AUTO preference and audio-source negotiation order.
 *
 * Stage C1 (`.accuracy-work/annotations/live-stage-c1-report.md`) measured Crema
 * ahead of ChordNet on real guitar (80.8 % vs 63.5 % family accuracy) and on the
 * Stage A synthetic piano grid, so AUTO must try Crema first.
 */
class LiveEngineSelectionTest {

    @Test
    fun `auto prefers crema then chordnet`() {
        assertEquals(
            listOf(ChordEngine.CREMA, ChordEngine.CHORD_NET),
            ChordEngine.AUTO_PREFERENCE,
        )
    }

    @Test
    fun `auto never selects the experimental or classic lanes`() {
        assertTrue(ChordEngine.BTC_EXPERIMENTAL !in ChordEngine.AUTO_PREFERENCE)
        assertTrue(ChordEngine.CLASSIC !in ChordEngine.AUTO_PREFERENCE)
        assertTrue(ChordEngine.AUTO !in ChordEngine.AUTO_PREFERENCE)
    }

    /**
     * solitito-ai is a guitar-only lane and AUTO must not reach for it.
     *
     * Stage C part 2 (`.accuracy-work/annotations/live-solitito-report.md`) put it
     * behind Crema pooled on real guitar and last of four on synthetic piano, where
     * it answers "single note, not a chord" for 99.3 % of a rootless voicing. It is
     * worth choosing deliberately for jazz and bossa comping and worth never
     * choosing by accident.
     */
    @Test
    fun `auto never selects solitito`() {
        assertTrue(ChordEngine.SOLITITO !in ChordEngine.AUTO_PREFERENCE)
        assertEquals(2, ChordEngine.AUTO_PREFERENCE.size)
    }

    @Test
    fun `chordnet and solitito stay explicitly selectable`() {
        assertTrue(ChordEngine.CHORD_NET in ChordEngine.entries)
        assertTrue(ChordEngine.SOLITITO in ChordEngine.entries)
    }

    @Test
    fun `unprocessed is tried first and still falls back to the shipped order`() {
        assertEquals(
            listOf(
                MediaRecorder.AudioSource.UNPROCESSED,
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                MediaRecorder.AudioSource.MIC,
            ),
            audioSourceCandidates(MediaRecorder.AudioSource.UNPROCESSED),
        )
    }

    @Test
    fun `the previous neural order is unchanged when unprocessed is not requested`() {
        assertEquals(
            listOf(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                MediaRecorder.AudioSource.MIC,
            ),
            audioSourceCandidates(MediaRecorder.AudioSource.VOICE_RECOGNITION),
        )
    }

    @Test
    fun `classic dsp keeps mic first`() {
        assertEquals(
            MediaRecorder.AudioSource.MIC,
            audioSourceCandidates(MediaRecorder.AudioSource.MIC).first(),
        )
    }

    @Test
    fun `source labels reach the backend string`() {
        assertEquals("unprocessed", audioSourceLabel(MediaRecorder.AudioSource.UNPROCESSED))
        assertEquals("voice-recognition", audioSourceLabel(MediaRecorder.AudioSource.VOICE_RECOGNITION))
        assertEquals("mic", audioSourceLabel(MediaRecorder.AudioSource.MIC))
        assertEquals("source-4242", audioSourceLabel(4242))
    }
}
