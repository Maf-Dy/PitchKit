package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.InstrumentProfile

/**
 * Pure offline DSP entry point. Samples must be mono PCM normalized to -1f..1f.
 * It deliberately skips temporal stabilization so tests can tell whether a bad
 * result comes from raw scoring or from realtime state.
 */
class PitchAnalyzer(
    private val profile: InstrumentProfile = InstrumentProfile.Guitar,
    private val sampleRate: Int = 44_100,
) {
    data class ChordFrame(
        val result: TuningResult.Chord?,
        val chroma: DoubleArray,
    )

    private val yin = YinPitchDetector(
        sampleRate = sampleRate,
        minFrequency = profile.minFreq,
        maxFrequency = profile.maxFreq,
    )
    private val chordDetector = ChordDetector(sampleRate, profile)

    fun analyzeNote(samples: FloatArray): TuningResult.Note? {
        val processed = FloatArray(samples.size)
        val rms = SignalPreprocessor.process(samples, processed)
        if (rms < profile.rmsGate) return null
        val frequency = yin.detect(processed)
        return NoteMapper.frequencyToNote(frequency, useFlats = profile.useFlats)?.let {
            TuningResult.Note(it.name, it.cents, it.frequency)
        }
    }

    fun analyzeChord(samples: FloatArray): ChordFrame {
        val processed = FloatArray(samples.size)
        val rms = SignalPreprocessor.process(samples, processed)
        if (rms < profile.rmsGate) return ChordFrame(null, DoubleArray(12))
        val analysis = chordDetector.analyze(processed)
        val result = analysis.chord?.let {
            TuningResult.Chord(it.name, it.score, it.confidence)
        }
        return ChordFrame(result, analysis.chroma)
    }
}
