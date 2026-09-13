package com.nicos.pitchkit.tuner

/** Selects the DSP path explicitly so note and chord modes avoid unnecessary work. */
enum class TunerMode {
    NOTE,
    CHORD,
    AUTO,
}
