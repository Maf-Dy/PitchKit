package com.nicos.pitchkit.tuner

/**
 * Chord-recognition backend used by [GuitarTunerListener] in CHORD mode.
 *
 * AUTO keeps the current production neural preference and never selects the
 * experimental BTC live path. Explicit neural choices fall back to Classic DSP
 * only when their assets are missing or the recognizer cannot be loaded.
 */
enum class ChordEngine {
    AUTO,
    CREMA,
    CHORD_NET,
    BTC_EXPERIMENTAL,
    CLASSIC,
}
