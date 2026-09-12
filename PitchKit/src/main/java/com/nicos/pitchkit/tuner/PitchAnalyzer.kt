package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.AudioFrame
import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlin.math.PI
import kotlin.math.sqrt

/** Source-agnostic pitch/chord analyzer. Feed normalized PCM from any accessible source. */
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
    private val history = ArrayDeque<String>()
    private var lastStable: TuningResult = TuningResult.Silence

    fun process(frame: AudioFrame): TuningResult {
        if (frame.samples.isEmpty()) return TuningResult.Silence
        ensureDetectors(frame.sampleRate)

        val buffer = preProcess(frame.toMono(), frame.sampleRate)
        if (buffer.size < 64) return TuningResult.Silence

        val energy = buffer.fold(0.0) { total, sample ->
            total + sample.toDouble() * sample.toDouble()
        } / buffer.size
        val rms = sqrt(energy)
        if (rms < profile.rmsGate) {
            chordDetector?.reset()
            return smooth(TuningResult.Silence)
        }

        val result = when (mode) {
            DetectionMode.NOTE -> detectNote(buffer)
            DetectionMode.CHORD -> detectChord(buffer)
            DetectionMode.AUTO -> detectAuto(buffer)
        }
        return smooth(result)
    }

    fun reset() {
        history.clear()
        lastStable = TuningResult.Silence
        chordDetector?.reset()
    }

    private fun detectNote(buffer: FloatArray): TuningResult {
        val frequency = yin?.detect(buffer) ?: return TuningResult.Silence
        return NoteMapper.frequencyToNote(
            freq = frequency,
            useFlats = profile.useFlats,
            referenceA4Hz = referenceA4Hz,
        )?.let { TuningResult.Note(it.name, it.cents, it.frequency) }
            ?: TuningResult.Silence
    }

    private fun detectChord(buffer: FloatArray): TuningResult =
        chordDetector?.detect(buffer, chordMinScore)
            ?.let { TuningResult.Chord(it.name) }
            ?: TuningResult.Silence

    private fun detectAuto(buffer: FloatArray): TuningResult {
        val detector = chordDetector ?: return TuningResult.Silence
        val strongPitchClasses = detector.chroma(buffer).count { it > autoChordThreshold }
        return if (strongPitchClasses <= 1) detectNote(buffer) else detectChord(buffer)
    }

    private fun ensureDetectors(sampleRate: Int) {
        if (sampleRate == detectorSampleRate && yin != null && chordDetector != null) return
        detectorSampleRate = sampleRate
        yin = YinPitchDetector(sampleRate)
        chordDetector = ChordDetector(sampleRate, profile, referenceA4Hz)
        reset()
    }

    private fun preProcess(raw: FloatArray, sampleRate: Int): FloatArray {
        if (raw.isEmpty()) return raw
        val output = raw.copyOf()
        val mean = output.average().toFloat()
        for (i in output.indices) output[i] -= mean
        if (highPassCutoffHz == 0.0) return output

        val dt = 1.0 / sampleRate
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

    private fun smooth(result: TuningResult): TuningResult {
        val key = when (result) {
            is TuningResult.Note -> "note:${result.name}"
            is TuningResult.Chord -> "chord:${result.name}"
            TuningResult.Silence -> "silence"
        }
        history.addLast(key)
        if (history.size > 4) history.removeFirst()

        val requiredAgreement = if (mode == DetectionMode.NOTE) 1 else 2
        if (history.count { it == key } >= requiredAgreement) lastStable = result
        return lastStable
    }
}
