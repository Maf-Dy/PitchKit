package com.nicos.pitchkit.tuner.harmony.solitito

import com.nicos.pitchkit.tuner.harmony.ChordRecognition
import com.nicos.pitchkit.tuner.harmony.ChordUpdateTracker
import com.nicos.pitchkit.tuner.harmony.ChordRecognizer
import com.nicos.pitchkit.tuner.models.AudioFrame

/**
 * The solitito-ai live lane: a guitar-trained chord classifier over a hand-built
 * pseudo-CQT front end.
 *
 * `greblus/solitito-ai` (MIT) ships a 7.4 M-parameter model whose ONNX graph takes
 * features rather than audio, so this class is mostly front end: resample to 16 kHz,
 * an 8192-point symmetric-Hann FFT every 256 samples, a sparse 4097 x 144 complex kernel,
 * chroma and bass folding ([SolititoFrontend]), then the app's own gate, vote and latch
 * ([SolititoDecisionLayer]).
 *
 * **What it is for.** Guitar, and specifically the seventh-chord repertoire. The Stage C
 * part 2 screening (`.accuracy-work/annotations/live-solitito-report.md`) put it at
 * 78.9 % family accuracy on the GuitarSet comp-60 slice against Crema's 80.8 %, but ahead
 * of Crema on Bossa Nova (82.6 % vs 77.6 %) and Jazz (79.0 % vs 78.0 %), with the best
 * latency profile of any lane measured (26.0 % of chord changes over 1.5 s against 31.1 %,
 * a 32 ms input bound) and by far the best extension evidence (tension-complete 32.0 %
 * against Crema's 12.3 %, and complete ≡ exact — it never covers by over-claiming).
 *
 * **What it is not for.** Piano, and anything without a root in the bass. It answers
 * "that is a single note, not a chord" for 99.3 % of a rootless voicing, 57.2 % of an open
 * tenth and 42.5 % of a chord rooted at C5, and it cannot hear a piano minor seventh
 * (17 % on `min7`, 0 % on `min13`). It is therefore an explicit choice only and is not in
 * `ChordEngine.AUTO_PREFERENCE`.
 *
 * **Two deliberate departures from upstream**, both documented where they are made:
 * the noise gate is [SolititoContract.GATE_DB] rather than the shipped −34 dBFS, and the
 * ORT session is not given upstream's thread setting. Nothing else is changed.
 *
 * **Compute.** One forward pass every 40 ms is 2.3x as often as ChordNet asks, for a
 * per-window cost that is actually the lowest of any neural lane (16.3 ms on the
 * screening desktop). If the duty cycle turns out to matter on a phone, the free lever is
 * [SolititoContract.INFERENCE_PERIOD_SECONDS]: relaxing 40 ms to ChordNet's 93 ms costs at
 * most one tick against a 717 ms median time-to-stable-label. That is a measured change
 * and is not made here.
 */
class SolititoStreamingRecognizer(
    modelBytes: ByteArray,
    planBytes: ByteArray,
    /**
     * The noise gate in dBFS on the raw frame RMS. Defaults to the port's
     * [SolititoContract.GATE_DB]; the replay harness passes
     * [SolititoContract.SHIPPED_GATE_DB] to reproduce the upstream ablation.
     */
    gateDb: Double = SolititoContract.GATE_DB,
    /** `lock_quality`, on by default upstream and here. */
    latchEnabled: Boolean = true,
) : ChordRecognizer {
    private val plan: SolititoDspPlan
    private val frontend: SolititoFrontend
    private val runner: SolititoOnnxRunner
    private val decisions: SolititoDecisionLayer
    private val resampler = SolititoLinearResampler()

    /** 16 kHz samples not yet consumed by a frame, oldest first. */
    private var pending = FloatArray(0)

    /** Absolute 16 kHz index of `pending[0]`, and of the next frame's first sample. */
    private var pendingBase = 0L
    private var nextFrameStart = 0L

    private val featureScratch = FloatArray(SolititoContract.FEATURE_COUNT)
    private val zeroFeatures = FloatArray(SolititoContract.FEATURE_COUNT)

    private var closed = false
    private val updates = ChordUpdateTracker()
    override val latestUpdate get() = updates.latest

    /**
     * The last decision this recognizer produced, in full.
     *
     * `ChordRecognition` carries only a label, a confidence and a backend, so the window
     * fill, the raw model reading, the vote and the latch state are invisible to anything
     * outside this class. They are the fields a live trace would want to attribute a
     * wrong label to the gate rather than the model, so they are kept here rather than
     * thrown away — module-internal, and read by nothing in `src/main`.
     */
    internal var lastDecision: SolititoDecisionLayer.Decision? = null
        private set

    init {
        require(SolititoContract.sha256(planBytes) == SolititoContract.PLAN_SHA256) {
            "Unexpected solitito DSP plan SHA-256"
        }
        plan = SolititoDspPlanDecoder.decodeAndVerify(planBytes)
        frontend = SolititoFrontend(plan, gateDb = gateDb)
        runner = SolititoOnnxRunner(modelBytes)
        decisions = SolititoDecisionLayer(
            model = { window -> runner.predict(window) },
            latchEnabled = latchEnabled,
        )
    }

    @Synchronized
    override fun recognize(frame: AudioFrame): ChordRecognition? {
        if (closed) return null
        val resampled = resampler.process(frame.toMono(), frame.sampleRate)
        if (resampled.isNotEmpty()) append(resampled)

        var emitted: ChordRecognition? = null
        while (available() >= SolititoContract.FFT_SIZE) {
            if (closed) return null
            val offset = (nextFrameStart - pendingBase).toInt()
            val rms = frontend.frameRms(pending, offset)
            val live = frontend.isLive(rms)
            val features = if (live) {
                frontend.features(pending, offset, featureScratch)
                featureScratch
            } else {
                zeroFeatures
            }
            val decision = decisions.push(features, live, rms)
            if (decision != null) {
                lastDecision = decision
                updates.record(decision.emitted?.let {ChordRecognition(it,decision.confidence,SolititoContract.BACKEND)})
                decision.emitted?.let {
                    emitted = ChordRecognition(
                        label = it,
                        confidence = decision.confidence,
                        backend = SolititoContract.BACKEND,
                    )
                }
            }
            nextFrameStart += SolititoContract.HOP_LENGTH
            dropConsumed()
        }
        return emitted
    }

    @Synchronized
    override fun reset() {
        if (closed) return
        resetState()
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        runner.close()
        resetState()
    }

    private fun resetState() {
        updates.reset()
        resampler.reset()
        pending = FloatArray(0)
        pendingBase = 0L
        nextFrameStart = 0L
        decisions.reset()
        lastDecision = null
    }

    private fun available(): Long = pendingBase + pending.size - nextFrameStart

    private fun append(samples: FloatArray) {
        val combined = FloatArray(pending.size + samples.size)
        pending.copyInto(combined)
        samples.copyInto(combined, destinationOffset = pending.size)
        pending = combined
    }

    private fun dropConsumed() {
        val drop = (nextFrameStart - pendingBase).toInt()
        if (drop <= 0) return
        pending = pending.copyOfRange(drop.coerceAtMost(pending.size), pending.size)
        pendingBase = nextFrameStart
    }
}
