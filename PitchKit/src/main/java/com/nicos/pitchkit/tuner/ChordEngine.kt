package com.nicos.pitchkit.tuner

/**
 * Chord-recognition backend used by [GuitarTunerListener] in CHORD mode.
 *
 * AUTO tries Crema first, then ChordNet, then the built-in classic DSP detector.
 * Explicit neural choices fall back to classic DSP only when their assets are missing
 * or the recognizer cannot be loaded, so audio capture still remains usable.
 */
enum class ChordEngine {
    AUTO,
    CREMA,
    CHORD_NET,
    CLASSIC,
}
