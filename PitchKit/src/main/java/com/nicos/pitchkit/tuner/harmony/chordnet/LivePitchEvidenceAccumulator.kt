package com.nicos.pitchkit.tuner.harmony.chordnet

import kotlin.math.max

/**
 * Keeps recently-heard pitch classes alive long enough for a normal arpeggio to
 * form one harmonic gesture. A settled chord can resolve quickly, while a chord
 * whose pitch set is still growing gets a bounded formation window.
 */
internal class LivePitchEvidenceAccumulator(
    private val decay: Float = 0.68f,
    private val activeThreshold: Float = 0.24f,
    private val minimumUpdates: Int = 2,
    private val forceReadyUpdates: Int = 5,
) {
    data class Snapshot(
        val pitch: FloatArray,
        val bass: FloatArray,
        /** Updates in the current decision window, not lifetime updates. */
        val updateCount: Int,
        val stableUpdates: Int,
        val ready: Boolean,
    )

    private val pitch = FloatArray(12)
    private val bass = FloatArray(12)
    private var lifetimeUpdates = 0
    private var decisionUpdates = 0
    private var stableUpdates = 0
    private var lastActiveMask = 0
    private var wasReady = false

    init {
        require(decay in 0f..1f)
        require(activeThreshold in 0f..1f)
        require(minimumUpdates >= 1)
        require(forceReadyUpdates >= minimumUpdates)
    }

    fun update(currentPitch: FloatArray, currentBass: FloatArray): Snapshot {
        require(currentPitch.size == 12)
        require(currentBass.size == 12)

        for (pc in 0 until 12) {
            pitch[pc] = max(pitch[pc] * decay, currentPitch[pc].coerceIn(0f, 1f))
            bass[pc] = max(bass[pc] * decay, currentBass[pc].coerceIn(0f, 1f))
        }
        normalizeInPlace(pitch)
        normalizeInPlace(bass)

        val activeMask = activeMask(currentPitch)
        val addedPitch = activeMask and lastActiveMask.inv()
        val pitchSetGrowing = lifetimeUpdates == 0 || addedPitch != 0

        if (lifetimeUpdates == 0) {
            decisionUpdates = 1
            stableUpdates = 0
        } else if (pitchSetGrowing) {
            // If a chord had already settled, a newly-arriving pitch starts a
            // fresh decision window. While an arpeggio is still forming, keep
            // counting from its first note so the maximum wait is truly bounded.
            decisionUpdates = if (wasReady) 1 else decisionUpdates + 1
            stableUpdates = 0
        } else {
            decisionUpdates++
            stableUpdates++
        }

        lastActiveMask = activeMask
        lifetimeUpdates++

        val ready = decisionUpdates >= minimumUpdates &&
            (stableUpdates >= 1 || decisionUpdates >= forceReadyUpdates)
        wasReady = ready

        return Snapshot(
            pitch = pitch.copyOf(),
            bass = bass.copyOf(),
            updateCount = decisionUpdates,
            stableUpdates = stableUpdates,
            ready = ready,
        )
    }

    fun reset() {
        pitch.fill(0f)
        bass.fill(0f)
        lifetimeUpdates = 0
        decisionUpdates = 0
        stableUpdates = 0
        lastActiveMask = 0
        wasReady = false
    }

    private fun activeMask(values: FloatArray): Int {
        var mask = 0
        for (pc in 0 until 12) {
            if (values[pc] >= activeThreshold) mask = mask or (1 shl pc)
        }
        return mask
    }

    private fun normalizeInPlace(values: FloatArray) {
        val peak = values.maxOrNull()?.coerceAtLeast(0f) ?: 0f
        if (peak <= 1e-8f) return
        for (index in values.indices) values[index] = (values[index] / peak).coerceIn(0f, 1f)
    }
}
