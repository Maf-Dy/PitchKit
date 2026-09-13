package com.nicos.pitchkit.tuner.extensions
// Modified in Maf-Dy/PitchKit fork: carry chord diagnostics through public API.

import com.nicos.pitchkit.tuner.TunerEngine
import com.nicos.pitchkit.tuner.TuningResult

internal fun TunerEngine.Result.toPublic(): TuningResult = when (this) {
    is TunerEngine.Result.Note -> TuningResult.Note(name = name, cents = cents, freq = freq)
    is TunerEngine.Result.Chord -> TuningResult.Chord(
        name = name,
        score = score,
        confidence = confidence,
    )
    TunerEngine.Result.Silence -> TuningResult.Silence
}
