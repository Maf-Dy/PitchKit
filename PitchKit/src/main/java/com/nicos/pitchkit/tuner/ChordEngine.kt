package com.nicos.pitchkit.tuner

/**
 * Chord-recognition backend used by [GuitarTunerListener] in CHORD mode.
 *
 * AUTO prefers **Crema**, falling back to ChordNet 2E1D when Crema's assets are
 * missing or its recognizer cannot be loaded, and never selects the experimental
 * BTC live path. The order follows the Stage C1 live benchmark
 * (`.accuracy-work/annotations/live-stage-c1-report.md`): on real guitar Crema
 * reached 80.8 % family accuracy against ChordNet's 63.5 % and Classic's 44.3 %,
 * and Crema also led ChordNet on the Stage A synthetic piano grid, so the
 * preference is not a trade of one instrument for another.
 *
 * Explicit neural choices — including [CHORD_NET], which stays selectable — fall
 * back to Classic DSP only when their assets are missing or the recognizer
 * cannot be loaded.
 */
enum class ChordEngine {
    AUTO,
    CREMA,
    CHORD_NET,
    /**
     * `greblus/solitito-ai` (MIT), a guitar-trained chord classifier over its own
     * pseudo-CQT front end. Deliberately **not** in [AUTO_PREFERENCE].
     *
     * The Stage C part 2 screening
     * (`.accuracy-work/annotations/live-solitito-report.md`) put it just behind Crema
     * pooled on real guitar (78.9 % against 80.8 %) but ahead of it on the two
     * seventh-chord styles — Bossa Nova 82.6 % against 77.6 %, Jazz 79.0 % against
     * 78.0 % — with the lowest latency bound of any lane and by far the best extension
     * evidence. It is also strictly guitar-shaped: it refuses to name a chord at all for
     * a rootless voicing or a chord above the guitar's register, and it cannot hear a
     * piano minor seventh. A player who wants it should choose it; Auto should not.
     */
    SOLITITO,
    BTC_EXPERIMENTAL,
    CLASSIC;

    companion object {
        /**
         * The neural lanes [AUTO] tries, best-measured first.
         *
         * Kept as data rather than an inline `?:` chain so the shipped preference
         * order is one testable fact instead of a detail buried in a composable.
         */
        val AUTO_PREFERENCE: List<ChordEngine> = listOf(CREMA, CHORD_NET)
    }
}
