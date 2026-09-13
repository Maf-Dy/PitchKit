package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.AudioFrame
import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlin.math.PI
import kotlin.math.sqrt

/**
 * Source-agnostic pitch/chord analyzer. Feed normalized PCM from any accessible source.
 *
 * Production callers should prefer an explicit [DetectionMode.NOTE] or
 * [DetectionMode.CHORD]. [DetectionMode.AUTO] is retained for compatibility only;
 * it necessarily guesses whether a frame is mono- or polyphonic.
 */
class PitchAnalyzer(
    val profile: InstrumentProfile = InstrumentProfile.Guitar,
    val mode: DetectionMode = DetectionMode.AUTO,
    val referenceA4Hz: Double = 440.0,
    val highPassCutoffHz: Double = 30.0,
    val autoChordThreshold: Double = 0.30,
    val chordMinScore: Double = 0.20,
) {
    init {
        require(referenceA4Hz > 0.0) { "referenceA4Hz must be > 0" }
        require(highPassCutoffHz >= 0.0) { "highPassCutoffHz must be >= 0" }
        require(autoChordThreshold in 0.0..1.0) { "autoChordThreshold must be between 0 and 1" }
    }

    private var detectorSampleRate = -1
    private var yin: YinPitchDetector? = null
    private var chordDetector: ChordDetector? = null
    private var silentFrames = 0
    private var preProcessScratch = FloatArray(0)

    fun process(frame: AudioFrame): TuningResult {
        if (frame.samples.isEmpty()) return TuningResult.Silence
        ensureDetectors(frame.sampleRate)

        val buffer = preProcess(frame.toMono(), frame.sampleRate)
        if (buffer.size < 64) return TuningResult.Silence

        var energy = 0.0
        for (sample in buffer) {
            val value = sample.toDouble()
            energy += value * value
        }
        val rms = sqrt(energy / buffer.size)

        if (rms < profile.rmsGate) {
            silentFrames++
            if (silentFrames >= 3) chordDetector?.reset()
            return TuningResult.Silence
        }

        silentFrames = 0
        return when (mode) {
            DetectionMode.NOTE -> detectNote(buffer)
            DetectionMode.CHORD -> detectChord(buffer)
            DetectionMode.AUTO -> detectAuto(buffer)
        }
    }

    fun reset() {
        silentFrames = 0
        chordDetector?.reset()
    }

    private fun detectNote(buffer: FloatArray): TuningResult {
        val frequency = yin?.detect(buffer) ?: return TuningResult.Silence
        return NoteMapper.frequencyToNote(
            freq = frequency,
            useFlats = profile.useFlats,
            referenceA4Hz = referenceA4Hz,
        )?.let {
            TuningResult.Note(it.name, it.cents, it.frequency)
        } ?: TuningResult.Silence
    }

    private fun detectChord(buffer: FloatArray): TuningResult =
        chordDetector?.detect(buffer, chordMinScore)
            ?.let {
                TuningResult.Chord(
                    name = it.name,
                    confidence = it.confidence,
                    backend = "Classic DSP",
                )
            }
            ?: TuningResult.Silence

    private fun detectAuto(buffer: FloatArray): TuningResult {
        val detector = chordDetector ?: return TuningResult.Silence
        val strongPitchClasses = detector.chroma(buffer).count { it > autoChordThreshold }
        return if (strongPitchClasses <= 1) detectNote(buffer) else detectChord(buffer)
    }

    private fun ensureDetectors(sampleRate: Int) {
        if (sampleRate == detectorSampleRate && yin != null && chordDetector != null) return
        detectorSampleRate = sampleRate
        yin = YinPitchDetector(
            sampleRate = sampleRate,
            minFrequencyHz = profile.minFreq,
            maxFrequencyHz = profile.maxFreq,
        )
        chordDetector = ChordDetector(sampleRate, profile, referenceA4Hz)
        reset()
    }

    private fun preProcess(raw: FloatArray, sampleRate: Int): FloatArray {
        if (raw.isEmpty()) return raw
        if (preProcessScratch.size != raw.size) {
            preProcessScratch = FloatArray(raw.size)
        }
        raw.copyInto(preProcessScratch)
        val output = preProcessScratch

        var mean = 0.0
        for (sample in output) mean += sample
        val meanFloat = (mean / output.size).toFloat()
        for (i in output.indices) output[i] -= meanFloat
        if (highPassCutoffHz == 0.0) return output

        val dt = 1.0 / sampleRate.toDouble()
        val rc = 1.0 / (2.0 * PI * highPassCutoffHz)
        val alpha = (rc / (rc + dt)).toFloat()
        var previousInput = 0f
        var previousOutput = 0f
        for (i in output.indices) {
            val input = output[i]
            val filtered = alpha * (previousOutput + input - previousInput)
            previousInput = input
            previousOutput = filtered
            output[i] = filtered
        }
        return output
    }
}
