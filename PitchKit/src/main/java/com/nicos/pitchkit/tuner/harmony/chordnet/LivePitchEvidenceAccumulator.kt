package com.nicos.pitchkit.tuner.harmony.chordnet

import kotlin.math.max

/**
 * Keeps recently-heard pitch classes alive long enough for normal arpeggios to
 * form one chord gesture instead of a sequence of partial chords.
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
        val updateCount: Int,
        val stableUpdates: Int,
        val ready: Boolean,
    )

    private val pitch = FloatArray(12)
    private val bass = FloatArray(12)
    private var updates = 0
    private var stableUpdates = 0
    private var lastActiveMask = 0

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
        stableUpdates = if (updates == 0 || addedPitch != 0) 0 else stableUpdates + 1
        lastActiveMask = activeMask
        updates++

        val ready = updates >= minimumUpdates &&
            (stableUpdates >= 1 || updates >= forceReadyUpdates)
        return Snapshot(
            pitch = pitch.copyOf(),
            bass = bass.copyOf(),
            updateCount = updates,
            stableUpdates = stableUpdates,
            ready = ready,
        )
    }

    fun reset() {
        pitch.fill(0f)
        bass.fill(0f)
        updates = 0
        stableUpdates = 0
        lastActiveMask = 0
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
