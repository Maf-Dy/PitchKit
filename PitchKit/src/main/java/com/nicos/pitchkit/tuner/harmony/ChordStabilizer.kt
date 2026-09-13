package com.nicos.pitchkit.tuner.harmony

/**
 * Separates backend classification from UI-facing chord stability.
 *
 * A backend may produce a different label for one inference while the player is
 * changing chords. During that transition we return null instead of pretending the
 * previous chord is still the current one. The caller can keep the last detected
 * chord on screen as historical UI state without reporting it as live audio.
 */
internal class ChordStabilizer(
    private val minimumConfidence: Double,
    private val changeConfirmations: Int = 2,
) {
    private var stable: ChordRecognition? = null
    private var pendingLabel: String? = null
    private var pendingCount = 0
    private var suppressStable = false

    init {
        require(minimumConfidence >= 0.0) { "minimumConfidence must be >= 0" }
        require(changeConfirmations >= 1) { "changeConfirmations must be >= 1" }
    }

    fun currentWithoutPrediction(): ChordRecognition? =
        if (suppressStable) null else stable

    fun update(prediction: ChordRecognition?): ChordRecognition? {
        if (prediction == null || prediction.confidence < minimumConfidence) {
            clearPending()
            suppressStable = stable != null
            return null
        }

        val current = stable
        if (current == null) {
            stable = prediction
            suppressStable = false
            clearPending()
            return stable
        }

        if (current.label == prediction.label) {
            stable = prediction
            suppressStable = false
            clearPending()
            return stable
        }

        suppressStable = true
        if (pendingLabel == prediction.label) {
            pendingCount++
        } else {
            pendingLabel = prediction.label
            pendingCount = 1
        }

        if (pendingCount >= changeConfirmations) {
            stable = prediction
            suppressStable = false
            clearPending()
            return stable
        }

        return null
    }

    fun reset() {
        stable = null
        suppressStable = false
        clearPending()
    }

    private fun clearPending() {
        pendingLabel = null
        pendingCount = 0
    }
}
