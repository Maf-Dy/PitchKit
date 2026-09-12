package com.nicos.pitchkit.tuner

import com.nicos.pitchkit.tuner.models.AudioFrame
import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlin.math.PI
import kotlin.math.sqrt

/**
 * Source-agnostic pitch/chord analyzer.
 *
 * This class has no dependency on AudioRecord or microphone permissions. Feed it
 * normalized PCM from a microphone, decoded file, USB interface, Media3 pipeline,
 * playback capture, or any other source that can provide PCM samples.
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

    private var detectorSampleRate: Int = -1
    private var yin: YinPitchDetector? = null
    private var chordDetector: ChordDetector? = null

    private val history = ArrayDeque<String>()
    private var lastStable: TuningResult = TuningResult.Silence
    private val requiredAgreement = 2

    /** Analyze one PCM frame and return the current stable result. */
    fun process(frame: AudioFrame): TuningResult {
        if (frame.samples.isEmpty()) return TuningResult.Silence

        ensureDetectors(frame.sampleRate)
        val buffer = preProcess(frame.toMono(), frame.sampleRate)
        if (buffer.isEmpty()) return TuningResult.Silence

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

    /** Clears temporal/hysteresis state when seeking or switching audio sources. */
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
        )?.let {
            TuningResult.Note(
                name = it.name,
                cents = it.cents,
                freq = it.frequency,
            )
        } ?: TuningResult.Silence
    }

    private fun detectChord(buffer: FloatArray): TuningResult {
        return chordDetector
            ?.detect(buffer, minScore = chordMinScore)
            ?.let { TuningResult.Chord(it.name) }
            ?: TuningResult.Silence
    }

    private fun detectAuto(buffer: FloatArray): TuningResult {
        val detector = chordDetector ?: return TuningResult.Silence
        val chroma = detector.chroma(buffer)
        val strongPitchClasses = chroma.count { it > autoChordThreshold }
        return if (strongPitchClasses <= 1) detectNote(buffer) else detectChord(buffer)
    }

    private fun ensureDetectors(sampleRate: Int) {
        if (sampleRate == detectorSampleRate && yin != null && chordDetector != null) return

        detectorSampleRate = sampleRate
        yin = YinPitchDetector(sampleRate)
        chordDetector = ChordDetector(
            sampleRate = sampleRate,
            profile = profile,
            referenceA4Hz = referenceA4Hz,
        )
        reset()
    }

    /** DC removal followed by a real low-cut filter (30 Hz by default). */
    private fun preProcess(raw: FloatArray, sampleRate: Int): FloatArray {
        if (raw.isEmpty()) return raw

        val out = raw.copyOf()
        val mean = out.average().toFloat()
        for (i in out.indices) out[i] -= mean

        if (highPassCutoffHz == 0.0) return out

        val dt = 1.0 / sampleRate.toDouble()
        val rc = 1.0 / (2.0 * PI * highPassCutoffHz)
        val alpha = (rc / (rc + dt)).toFloat()

        var previousInput = 0f
        var previousOutput = 0f
        for (i in out.indices) {
            val input = out[i]
            val output = alpha * (previousOutput + input - previousInput)
            previousInput = input
            previousOutput = output
            out[i] = output
        }
        return out
    }

    /** Require repeated agreement before exposing a new stable result. */
    private fun smooth(result: TuningResult): TuningResult {
        val key = when (result) {
            is TuningResult.Note -> "note:${result.name}"
            is TuningResult.Chord -> "chord:${result.name}"
            TuningResult.Silence -> "silence"
        }

        history.addLast(key)
        if (history.size > 4) history.removeFirst()

        val agreement = history.count { it == key }
        if (agreement >= requiredAgreement) {
            lastStable = result
        }
        return lastStable
    }
}
