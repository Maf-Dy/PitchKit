package com.nicos.pitchkit.tuner
// Modified in Maf-Dy/PitchKit fork: DSP correctness, performance, and lifecycle fixes.

import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn

private fun defaultFrameSize(mode: TunerMode): Int = when (mode) {
    TunerMode.NOTE -> 4096
    TunerMode.CHORD, TunerMode.AUTO -> 8192
}

internal class TunerEngine(
    private val profile: InstrumentProfile = InstrumentProfile.Guitar,
    private val mode: TunerMode = TunerMode.AUTO,
    sampleRate: Int = 44_100,
    bufferSize: Int = defaultFrameSize(mode),
) {
    private val capture = AudioCapture(sampleRate, bufferSize)
    private val yin = YinPitchDetector(
        sampleRate = sampleRate,
        minFrequency = profile.minFreq,
        maxFrequency = profile.maxFreq,
    )
    private val chordDetector = ChordDetector(sampleRate, profile)
    private val chordStabilizer = ChordStabilizer()
    private val processed = FloatArray(bufferSize)

    sealed class Result {
        data class Note(val name: String, val cents: Double, val freq: Float) : Result()
        data class Chord(
            val name: String,
            val score: Double,
            val confidence: Double,
        ) : Result()
        object Silence : Result()
    }

    fun start() = callbackFlow<Result> {
        capture.start(scope = this) { raw ->
            val rms = SignalPreprocessor.process(raw, processed)
            if (rms < profile.rmsGate) {
                chordStabilizer.reset()
                trySend(Result.Silence)
                return@start
            }

            val result = when (mode) {
                TunerMode.NOTE -> detectNote(processed)
                TunerMode.CHORD -> detectChord(processed)
                TunerMode.AUTO -> detectAuto(processed)
            }
            trySend(result)
        }
        awaitClose { stop() }
    }
        .buffer(capacity = Channel.CONFLATED)
        .flowOn(Dispatchers.Default)

    fun stop() {
        chordStabilizer.reset()
        capture.stop()
    }

    private fun detectNote(buffer: FloatArray): Result {
        val frequency = yin.detect(buffer)
        return NoteMapper.frequencyToNote(frequency, useFlats = profile.useFlats)?.let {
            Result.Note(it.name, it.cents, it.frequency)
        } ?: Result.Silence
    }

    private fun detectChord(buffer: FloatArray): Result {
        val stable = chordStabilizer.accept(chordDetector.detect(buffer))
        return stable?.let { Result.Chord(it.name, it.score, it.confidence) }
            ?: Result.Silence
    }

    private fun detectAuto(buffer: FloatArray): Result {
        val analysis = chordDetector.analyze(buffer)
        val strongPitchClasses = analysis.chroma.count { it > 0.50 }
        return if (strongPitchClasses <= 1) {
            chordStabilizer.reset()
            detectNote(buffer)
        } else {
            val stable = chordStabilizer.accept(analysis.chord)
            stable?.let { Result.Chord(it.name, it.score, it.confidence) }
                ?: Result.Silence
        }
    }
}
