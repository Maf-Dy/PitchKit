package com.nicos.pitchkit.tuner.harmony.solitito

/**
 * Everything between the 168 features and the label on screen — `code/src/main.rs` plus
 * `code/src/latch.rs`, with the gate constant of [SolititoContract.GATE_DB].
 *
 * Per analysis frame (62.5 a second) it advances the envelope attack detector and the
 * rolling 48-frame history. Every 40 ms it asks the model, provided the history is at
 * least [SolititoContract.MIN_FILL] live; the answer may name a chord only if the history
 * is at least [SolititoContract.MIN_FILL_CHORD] live. Named answers go through a
 * confidence-weighted majority vote over the last three of them and then through
 * [SolititoChordLatch].
 *
 * Kept apart from [SolititoStreamingRecognizer] because it is the half that is pure: it
 * takes features and an RMS and returns a decision, so `SolititoParityTest` can drive it
 * straight from the Python fixture's own features and diff the emitted label sequence
 * without an ONNX session or a WAV file in the way.
 */
internal class SolititoDecisionLayer(
    private val model: Model,
    latchEnabled: Boolean = true,
    private val contextFrames: Int = SolititoContract.CONTEXT_FRAMES,
) {
    /** The inference seam: 48 x 168 features in, one reading out. */
    fun interface Model {
        fun predict(window: FloatArray): Reading
    }

    /** `brain.rs::predict` — argmax plus softmax on both heads. */
    data class Reading(
        val rootIndex: Int,
        val quality: String,
        val confidence: Double,
        /** Twelve sigmoid pitch activations; carried for diagnostics, not decided on. */
        val pitches: FloatArray? = null,
    ) {
        /** The internal chord identity: `"Noise"`, `"Note C"`, `"C"`, `"C m7"`, ... */
        val chord: String get() = SolititoVocabulary.chordString(rootIndex, quality)
    }

    /** One inference tick's worth of state, everything the trace and the tests want. */
    data class Decision(
        val frameIndex: Long,
        /** Fraction of the 48-frame history that cleared the gate. */
        val fill: Double,
        /** False when the window was too empty to even ask. */
        val asked: Boolean,
        /** The label the screen shows, or `null` when nothing new was emitted. */
        val emitted: String?,
        val confidence: Double,
        val rawChord: String?,
        val modelConfidence: Double?,
        val quality: String?,
        val voted: String?,
        val latched: String?,
        val onsetId: Int,
    )

    private val latch = SolititoChordLatch(latchEnabled)

    private val history = FloatArray(contextFrames * SolititoContract.FEATURE_COUNT)
    private val historyLive = BooleanArray(contextFrames)
    private var writeIndex = 0
    private var liveCount = 0

    private val window = FloatArray(contextFrames * SolititoContract.FEATURE_COUNT)

    private val votedChords = ArrayDeque<String>()
    private val votedScores = ArrayDeque<Double>()

    private var frameIndex = 0L
    private var onsetId = 0
    private var framesSinceOnset = 0
    private var framesSinceAttack = SolititoContract.ATTACK_REFRACTORY
    private var baseline = 0.0
    private var stride = 1
    private var ticksSinceInference = 0

    /** How many 40 ms ticks pass between forward passes right now (1 = every tick). */
    val currentStride: Int get() = stride

    private fun adaptStride(elapsedMs: Double) {
        val budgetMs = SolititoContract.INFERENCE_PERIOD_SECONDS * 1000.0 * stride
        if (elapsedMs > budgetMs * SLOW_FRACTION && stride < MAX_STRIDE) {
            stride++
        } else if (stride > 1 && elapsedMs < budgetMs * FAST_FRACTION) {
            stride--
        }
    }

    // ------------------------------------------------------------- diagnostics
    var inferences = 0L
        private set
    var skippedLowFill = 0L
        private set
    var notNamed = 0L
        private set
    var noteReadings = 0L
        private set
    var noiseReadings = 0L
        private set
    var gatedFrames = 0L
        private set
    var liveFrames = 0L
        private set

    /**
     * Advance one analysis frame.
     *
     * [features] must be 168 values, all zero when the frame is gated; [live] is whether
     * it cleared the gate; [rms] is the raw frame RMS, which the attack detector reads
     * whether the frame is gated or not.
     *
     * Returns a [Decision] on the 40 ms ticks and `null` between them.
     */
    fun push(features: FloatArray, live: Boolean, rms: Float): Decision? {
        require(features.size == SolititoContract.FEATURE_COUNT) {
            "features must be ${SolititoContract.FEATURE_COUNT} values"
        }
        advanceEnvelope(rms.toDouble())
        appendHistory(features, live)
        if (live) liveFrames++ else gatedFrames++

        val index = frameIndex
        frameIndex++
        if (!SolititoContract.isInferenceFrame(index)) return null
        // Adaptive cadence: a phone that cannot finish one forward pass inside the 40 ms
        // tick drops capture audio instead (18 % of buffers on the first device run).
        // Skipping ticks keeps the audio continuous; the three-window vote then spans
        // 80-160 ms instead of 120 ms, which the latch tolerates.
        ticksSinceInference++
        if (ticksSinceInference < stride) return null
        ticksSinceInference = 0

        val fill = liveCount.toDouble() / contextFrames
        if (fill < SolititoContract.MIN_FILL) {
            // The inference thread does not even ask; the screen keeps what it had.
            skippedLowFill++
            return Decision(
                frameIndex = index, fill = fill, asked = false, emitted = null,
                confidence = 0.0, rawChord = null, modelConfidence = null, quality = null,
                voted = null, latched = latch.locked, onsetId = onsetId,
            )
        }

        val startedNanos = System.nanoTime()
        val reading = model.predict(copyWindow())
        adaptStride((System.nanoTime() - startedNanos) / 1_000_000.0)
        inferences++
        val chord = reading.chord
        if (reading.quality == "note") noteReadings++
        if (chord == SolititoVocabulary.NOISE) noiseReadings++

        val named = fill >= SolititoContract.MIN_FILL_CHORD
        if (!named) notNamed++
        if (named) pushVote(chord, reading.confidence)

        // Confidence-weighted majority over the last three named windows. The tally is
        // insertion-ordered and the comparison is strictly greater, so the first chord
        // inserted wins a tie — and a history of nothing but `Noise`, whose confidence is
        // always 0.0, never beats the 0.0 floor and leaves the vote unresolved.
        val tally = LinkedHashMap<String, Double>()
        for (position in votedChords.indices) {
            val name = votedChords[position]
            tally[name] = (tally[name] ?: 0.0) + votedScores[position]
        }
        var best = SolititoVocabulary.PENDING
        var bestScore = 0.0
        for ((name, score) in tally) {
            if (score > bestScore) {
                best = name
                bestScore = score
            }
        }
        val votedConfidence = if (votedChords.isEmpty()) 0.0 else bestScore / votedChords.size

        val shown: String
        val emittedConfidence: Double
        if (named) {
            shown = latch.update(onsetId, framesSinceOnset, best, votedConfidence)
            emittedConfidence = votedConfidence
        } else {
            shown = SolititoVocabulary.PENDING
            emittedConfidence = 0.0
        }

        return Decision(
            frameIndex = index,
            fill = fill,
            asked = true,
            emitted = SolititoVocabulary.toDisplayLabel(shown),
            confidence = emittedConfidence,
            rawChord = chord,
            modelConfidence = reading.confidence,
            quality = reading.quality,
            voted = best,
            latched = latch.locked,
            onsetId = onsetId,
        )
    }

    fun reset() {
        stride = 1
        ticksSinceInference = 0
        java.util.Arrays.fill(history, 0f)
        java.util.Arrays.fill(historyLive, false)
        writeIndex = 0
        liveCount = 0
        votedChords.clear()
        votedScores.clear()
        latch.reset()
        frameIndex = 0L
        onsetId = 0
        framesSinceOnset = 0
        framesSinceAttack = SolititoContract.ATTACK_REFRACTORY
        baseline = 0.0
        inferences = 0L
        skippedLowFill = 0L
        notNamed = 0L
        noteReadings = 0L
        noiseReadings = 0L
        gatedFrames = 0L
        liveFrames = 0L
    }

    /** The envelope attack detector from the stream callback in `audio.rs`. */
    private fun advanceEnvelope(level: Double) {
        if (level > baseline * SolititoContract.ATTACK_RATIO &&
            level > SolititoContract.ATTACK_FLOOR &&
            framesSinceAttack >= SolititoContract.ATTACK_REFRACTORY
        ) {
            onsetId++
            framesSinceOnset = 0
            framesSinceAttack = 0
        } else {
            framesSinceAttack++
        }
        baseline = if (level > baseline) {
            baseline * 0.90 + level * 0.10
        } else {
            baseline * 0.70 + level * 0.30
        }
        framesSinceOnset++
    }

    private fun appendHistory(features: FloatArray, live: Boolean) {
        if (historyLive[writeIndex]) liveCount--
        historyLive[writeIndex] = live
        if (live) liveCount++
        features.copyInto(history, destinationOffset = writeIndex * SolititoContract.FEATURE_COUNT)
        writeIndex = (writeIndex + 1) % contextFrames
    }

    /** Oldest frame first, which is the order the model's time axis expects. */
    private fun copyWindow(): FloatArray {
        val stride = SolititoContract.FEATURE_COUNT
        for (position in 0 until contextFrames) {
            val source = (writeIndex + position) % contextFrames
            history.copyInto(
                destination = window,
                destinationOffset = position * stride,
                startIndex = source * stride,
                endIndex = source * stride + stride,
            )
        }
        return window
    }

    private fun pushVote(chord: String, confidence: Double) {
        votedChords.addLast(chord)
        votedScores.addLast(confidence)
        while (votedChords.size > SolititoContract.VOTE_WINDOWS) {
            votedChords.removeFirst()
            votedScores.removeFirst()
        }
    }

    private companion object {
        /** Never coarser than one forward pass per 160 ms of audio. */
        const val MAX_STRIDE = 4
        /** A pass that used more than this share of its tick budget is "too slow". */
        const val SLOW_FRACTION = 0.85
        /** A pass that used less than this share can afford a finer cadence again. */
        const val FAST_FRACTION = 0.35
    }
}
