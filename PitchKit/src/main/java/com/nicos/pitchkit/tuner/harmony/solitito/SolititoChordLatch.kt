package com.nicos.pitchkit.tuner.harmony.solitito

/**
 * `code/src/latch.rs` — a chord does not change identity while it rings out.
 *
 * Once a four-note chord has settled, the latch holds it through its own decay (a `C m7`
 * whose seventh has died away reads as `C m`, or even as a single note, and neither may
 * rewrite the label) and releases only on a new attack. A confident reading of a genuinely
 * different chord still gets through immediately, so the latch delays nothing real.
 *
 * `lock_quality` is on by default upstream and is on here. The screening measured it and
 * the vote together as worth their place: with the gate opened, the full pipeline matches
 * its own raw model's family accuracy (78.9 % against 78.3 %) while cutting flicker from
 * 6.75 to 2.11 changes per segment and extended-label flicker from 1 217 to 194.
 */
internal class SolititoChordLatch(private val enabled: Boolean = true) {
    var locked: String? = null
        private set

    private var lastOnset = 0

    fun update(onsetId: Int, framesSinceOnset: Int, chord: String, confidence: Double): String {
        if (!enabled) {
            locked = null
            lastOnset = onsetId
            return chord
        }
        if (onsetId != lastOnset) {
            lastOnset = onsetId
            locked = null
        }
        val held = locked
        if (held != null) {
            if (SolititoVocabulary.isDecayOf(held, chord)) return held
            if (confidence >= SolititoContract.LOCK_MIN_CONFIDENCE &&
                SolititoVocabulary.isRealChord(chord)
            ) {
                locked = chord
                return chord
            }
            return held
        }
        if (framesSinceOnset >= SolititoContract.SETTLE_FRAMES &&
            confidence >= SolititoContract.LOCK_MIN_CONFIDENCE &&
            SolititoVocabulary.isRealChord(chord)
        ) {
            locked = chord
        }
        return chord
    }

    fun reset() {
        locked = null
        lastOnset = 0
    }
}
