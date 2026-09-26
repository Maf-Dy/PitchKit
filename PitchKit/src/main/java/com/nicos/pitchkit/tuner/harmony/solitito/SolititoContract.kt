package com.nicos.pitchkit.tuner.harmony.solitito

import java.security.MessageDigest
import kotlin.math.exp

/**
 * Every constant the solitito-ai live lane is pinned to.
 *
 * The upstream project (`greblus/solitito-ai`, MIT, archived under
 * `.accuracy-work/references/solitito-ai`) serves a 7.4 M-parameter chord model whose
 * ONNX graph takes **features, not audio**: its only input is `features[1, 48, 168]`.
 * The whole port cost is therefore the front end, and these numbers are that front end's
 * contract, transcribed from `code/src/audio.rs`, `code/src/main.rs`, `code/src/brain.rs`
 * and `code/src/latch.rs` by way of the validated Python reference
 * `tools/accuracy_audit/solitito_live.py`, which is the spec this port is tested against.
 *
 * One constant is **deliberately not** the shipped one. See [GATE_DB].
 */
object SolititoContract {
    // ------------------------------------------------------------------ geometry

    /** Everything is resampled to 16 kHz before the first FFT (`audio.rs`). */
    const val SAMPLE_RATE = 16_000

    /**
     * 256 samples at 16 kHz: a 16 ms frame period, 62.5 analysis frames a second.
     *
     * This is also the capture buffer `GuitarTunerListener` asks `AudioCapture` to
     * negotiate, which makes solitito's input latency term (`2 x 256 / 16000` = 32 ms)
     * the smallest of any lane in the APK.
     */
    const val HOP_LENGTH = 256

    /**
     * Samples per capture buffer on the device: 64 ms, four hops. One forward pass on a
     * phone takes longer than a single 16 ms hop, and the capture channel is conflated,
     * so a buffer shorter than one inference is dropped audio (18 % of buffers on the
     * first device run). The recognizer consumes hop by hop regardless of buffer size.
     */
    const val LIVE_BUFFER_SAMPLES = 4 * HOP_LENGTH

    /** One 8192-point FFT per frame: a 512 ms analysis window at every bin. */
    const val FFT_SIZE = 8_192

    /** `rfft` output length. */
    const val FFT_BINS = FFT_SIZE / 2 + 1

    /** The model's context: 48 frames, i.e. 47 hops + one window = 1.264 s of lookback. */
    const val CONTEXT_FRAMES = 48

    const val CQT_BINS = 144
    const val CHROMA_BINS = 12
    const val BASS_BINS = 12

    /** `concat(cqt144, chroma12, bass12)` — the model's 168 inputs per frame. */
    const val FEATURE_COUNT = CQT_BINS + CHROMA_BINS + BASS_BINS

    // ------------------------------------------------------------------ the DSP

    /** `INPUT_GAIN` applied after the gate and before the window (`audio.rs`). */
    const val INPUT_GAIN = 2.0f

    /** `bass_boost_enabled: true` / `bass_boost_gain: 5.0` (`state.rs:587-588`). */
    const val BASS_BOOST_ENABLED = true
    const val BASS_BOOST_GAIN = 5.0
    const val BASS_BOOST_CUTOFF = 36

    /** Floor under the per-frame log-normalisation reference. */
    const val MIN_REFERENCE_LEVEL = 0.005

    /** The per-frame normalisation spans 80 dB below the frame's own peak. */
    const val NORMALISATION_DB = 80.0

    // ------------------------------------------------------------------ the gate

    /**
     * The noise gate, in dBFS on the **raw** (pre-[INPUT_GAIN]) frame RMS.
     *
     * Upstream ships `default_gate_db = -34.0` (`main.rs:750`). This port does not.
     * The Stage C part 2 screening (`.accuracy-work/annotations/live-solitito-report.md`
     * §5) measured that constant as the model's single largest defect: because the gate
     * sits *in front of* the 48-frame history and [MIN_FILL_CHORD] refuses to name a
     * chord unless the whole 768 ms of history is above it, the screen blanks for the
     * second half of every sustained chord while the model is still reading it correctly.
     * Moving the gate to −60 dBFS and changing nothing else was worth **+15.1 points** of
     * family accuracy on real guitar (63.8 % → 78.9 %) and cut blank-screen frames from
     * 21.3 % to 0.8 %. The author's own source carries the diagnosis
     * (`audio.rs::history_fill`: *"That is why chords appeared to resolve only in the
     * tail"*); the constant was simply never moved.
     *
     * `TunerEngine` has its own RMS gate in front of the recognizer, so nothing is lost
     * by opening this one.
     */
    const val GATE_DB = -60.0

    /** The upstream default, kept as data so the ablation is one argument away. */
    const val SHIPPED_GATE_DB = -34.0

    // ------------------------------------------------- main.rs / latch.rs decisions

    /** The 48-frame window must be at least half live before the model is asked at all. */
    const val MIN_FILL = 0.5

    /** ...and 90 % live before its answer is allowed to name a chord. */
    const val MIN_FILL_CHORD = 0.9

    /**
     * One forward pass every 40 ms of audio.
     *
     * 40 ms is 2.5 frames at a 16 ms hop, so the tick lands on frame indices
     * `floor(2.5k)` = 0, 2, 5, 7, 10, 12, … — exactly the frames whose index modulo 5
     * is 0 or 2. [isInferenceFrame] is that test and `SolititoDecisionLayerTest` pins it
     * against the general formula.
     */
    const val INFERENCE_PERIOD_SECONDS = 0.040

    /** Confidence-weighted majority vote over this many *named* windows. */
    const val VOTE_WINDOWS = 3

    /** `latch.rs`: the confidence a reading needs before it may take or hold the lock. */
    const val LOCK_MIN_CONFIDENCE = 0.60

    /** `latch.rs`: frames after an attack before the latch may close. */
    const val SETTLE_FRAMES = 48

    /** The envelope attack detector in the stream callback (`audio.rs`). */
    const val ATTACK_RATIO = 1.8
    const val ATTACK_FLOOR = 0.01
    const val ATTACK_REFRACTORY = 12

    // ------------------------------------------------------------------ the model

    const val INPUT_NAME = "features"
    const val ROOT_OUTPUT = "root_logits"
    const val QUALITY_OUTPUT = "quality_logits"
    const val PITCH_OUTPUT = "pitch_logits"

    const val ROOT_CLASSES = 13
    const val QUALITY_CLASSES = 11
    const val PITCH_CLASSES = 12

    /** What the live screen shows as the backend for this lane. */
    const val BACKEND = "solitito-ai take6 · live"

    // ------------------------------------------------------------------ the assets

    const val ASSET_DIRECTORY = "solitito"
    const val MODEL_FILE = "solitito-v2-take6.onnx"
    const val PLAN_FILE = "solitito-dsp.bin"

    /** `best_model_v2_take6.onnx`, 29 291 803 B, from `greblus/solitito-ai` on the Hub. */
    const val MODEL_SHA256 = "e2c8d451330b43cb8b1ad8560a315ea36427a091f254ab9390b37b13d171786b"

    /**
     * `solitito-dsp.bin`, 495 720 B, generated from the upstream `dsp_weights.json`
     * (2 119 975 B, SHA-256 `26fd0135…`) by
     * `tools/accuracy_audit/export_solitito_assets.py`. The JSON itself is not shipped.
     */
    const val PLAN_SHA256 = "566d4a3c0ad9e1907064269e302ad4968e14a204e8f8a245298113535e667c2b"

    /** SHA-256 of the upstream JSON the plan was packed from, recorded for provenance. */
    const val UPSTREAM_WEIGHTS_SHA256 =
        "26fd0135195a4e55cf2791f4d9e6275f1ba2dc36e1b0d2d5121836ca4633a823"

    /** `greblus/solitito-ai`, branch `solitito-ai`. */
    const val SOURCE_COMMIT = "57a433db6c945d71f70037e9c9dd08b2865ef848"

    // ------------------------------------------------------------------ helpers

    /** Linear amplitude for a dBFS threshold. */
    fun dbToLinear(db: Double): Double = Math.pow(10.0, db / 20.0)

    /**
     * Whether analysis frame [frameIndex] is one of the 25-a-second inference ticks.
     *
     * `int(step * 2.5)` over the integers is `{0, 2, 5, 7, 10, 12, …}`, which is exactly
     * `frameIndex % 5 in {0, 2}`.
     */
    fun isInferenceFrame(frameIndex: Long): Boolean {
        val phase = (frameIndex % 5L).toInt()
        return phase == 0 || phase == 2
    }

    /**
     * `brain.rs`: the argmax of a head and its softmax probability.
     *
     * The argmax is the *first* maximum, which is `np.argmax`'s rule and therefore the
     * reference's; the exponentials are accumulated in `Double` after subtracting the
     * peak, which is the reference's too. It lives here rather than beside the ONNX
     * session so the parity test can replay the reference's own logits through the
     * decision layer without loading ONNX Runtime's native library.
     */
    fun argmaxSoftmax(logits: FloatArray): Pair<Int, Double> {
        require(logits.isNotEmpty()) { "an empty logit vector has no argmax" }
        var index = 0
        var peak = logits[0]
        for (position in logits.indices) {
            if (logits[position] > peak) {
                peak = logits[position]
                index = position
            }
        }
        var total = 0.0
        for (value in logits) total += exp(value.toDouble() - peak.toDouble())
        return index to (exp(logits[index].toDouble() - peak.toDouble()) / total)
    }

    fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
