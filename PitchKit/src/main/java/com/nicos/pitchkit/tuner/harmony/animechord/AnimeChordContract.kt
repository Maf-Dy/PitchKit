package com.nicos.pitchkit.tuner.harmony.animechord

import kotlin.math.ln

/**
 * Asset and graph contract for the offline `anime Chord-Transcription` song engine.
 *
 * anime-song/Chord-Transcription (github.com/anime-song/Chord-Transcription, HF revision
 * `acac9895`, checkpoint `model_epoch_150_public.pt` — the only public checkpoint any
 * published code can load) is a 5.4 M-parameter RoPE time transformer over a recursive
 * constant-Q front end, with a 745-class root-chord head (12 roots x 62 dlchordx
 * qualities, plus N), a 13-class bass head and a 13-class key head. It is the only engine
 * in this app that ever names a ninth.
 *
 * Three things fix the shape of the port, all measured in
 * `.accuracy-work/annotations/anime-windowed-report.md`:
 *
 *  * ONNX Runtime materialises the attention matrices torch's memory-efficient SDPA
 *    avoids, so peak native memory grows quadratically with the sequence: 220 MB at 15 s,
 *    455 MB at 30 s, 910 MB at 45 s, 9.3 GB at 240 s. Whole-song inference is impossible
 *    on a phone; the model must be windowed.
 *  * Windowing does not cost accuracy — it buys it. The base model was trained on 60 s
 *    segments (`configs/train_large.yaml`), and whole-song inference runs its time
 *    transformer up to twelve times past the longest sequence it ever saw. Every
 *    configuration of the `{15,30,45} s x {0,25,50} %` grid beat the whole-song lane on
 *    root, family, exact and boundary F1.
 *  * [WINDOW_SECONDS] / [OVERLAP_PERCENT] is the grid's best on `exact` (56.2) and
 *    boundary F1 (68.8) and its lead there clears a song-level bootstrap.
 *
 * Window 0 is **not** primed — it starts at real frame 0 and commits from its own frame 0,
 * the same lesson `chordformer-phone-diagnosis.md` measured for ChordFormer.
 *
 * Assets are produced by `tools/generate-animechord-cqt-plans.py`, which reads the
 * kernels out of the reference `RecursiveCQT`'s own buffers and the qualities out of the
 * checkpoint's own `label_vocab`, so nothing here is hand-written harmony.
 */
internal object AnimeChordContract {
    const val ASSET_DIRECTORY = "animechord"
    const val METADATA_FILE = "animechord-meta.json"
    const val MODEL_FILE = "anime-chord-from-spec.onnx"
    const val CQT_PLAN_FILE = "animechord-cqt.bin"
    const val VOCABULARY_FILE = "animechord-vocab.json"

    const val SAMPLE_RATE = 22_050
    const val HOP_LENGTH = 512
    const val BINS_PER_OCTAVE = 36
    const val INPUT_BINS = 252
    const val OCTAVES = INPUT_BINS / BINS_PER_OCTAVE
    const val STAGE_FFT = 256
    const val STAGE_BINS = STAGE_FFT / 2 + 1
    const val DECIMATOR_TAPS = 65
    const val FMIN_HZ = 32.7
    const val FILTER_SCALE = 0.4375

    /**
     * `AudioFeatureExtractor` derives `crop_length = (samples - 2048) / 512 + 1` so the
     * frame grid lines up with a `center=True` STFT at that size. The exported graph
     * makes the trim an identity, so the host does it.
     */
    const val CROP_N_FFT = 2048

    /** `(x - mean) / (std + 1e-8)` per channel over the whole (freq, time) block. */
    const val STANDARDISATION_EPSILON = 1e-8

    const val WINDOW_SECONDS = 30.0
    const val OVERLAP_PERCENT = 50

    /** 30.0 s at 43.066 frames/s. */
    const val WINDOW_FRAMES = 1292
    const val OVERLAP_FRAMES = 646
    const val HALF_OVERLAP_FRAMES = OVERLAP_FRAMES / 2

    /** The graph pads its own time axis to a multiple of eight before up-sampling. */
    const val TIME_MULTIPLE = 8

    const val ROOT_CHORD_COUNT = 745
    const val BASS_COUNT = 13
    const val KEY_COUNT = 13
    const val QUALITY_COUNT = 62
    const val ROOT_COUNT = 12

    const val INPUT_NAME = "spec"
    const val ROOT_CHORD_OUTPUT = "root_chord"
    const val BASS_OUTPUT = "bass"
    const val KEY_OUTPUT = "key"
    const val BOUNDARY_OUTPUT = "boundary"
    const val BEAT_OUTPUT = "beat"
    const val DOWNBEAT_OUTPUT = "downbeat"

    /**
     * The author's HMM is `make_sticky_transition(C, stay_prob)` — `stay_prob` on the
     * diagonal, `(1 - stay_prob) / (C - 1)` uniform everywhere else — fed to a Viterbi
     * over `log(softmax(logits) + 1e-9)` with a uniform initial distribution. At the
     * author's `stay_prob = 1.0` for the root-chord and bass heads, `log(A + 1e-9)` is
     * exactly 0 on the diagonal and `ln(1e-9) = -20.723` off it: a constant switch
     * penalty, not a learned transition model.
     */
    const val LOG_STAY = 0.0
    val LOG_SWITCH: Double = ln(1e-9)
    const val EMISSION_EPSILON = 1e-9

    /** `PredictionDecoder.to_events(min_duration_chord = 0.1)`. */
    const val MIN_CHORD_SECONDS = 0.1

    const val CHECKPOINT = "model_epoch_150_public.pt"
    const val CHECKPOINT_SHA256 =
        "46cc4a6d232659f53f89f09b8a9fca01771715c818955f844867b0944ffe1fb8"
    const val HF_REVISION = "acac989522bf025848d8cec754b78e5f20a2e810"
    const val SOURCE_COMMIT = "c1b029de45caff58b4d62586407b31cb4c32cad8"
    const val MODEL_SHA256 =
        "b372c6550b00ff71b42721e462c6f0fef08b49c6a5c121334f1aadb0f1dbdfd3"
    const val CQT_PLAN_SHA256 =
        "36fec025a5e0ba287129f2aebab5c598cafd79f5280e6c31caffee8aad2ee3c4"
    const val VOCABULARY_SHA256 =
        "f6787d5ac58c51ce74b7f360d1787b6b928ee4d95cdb2bc9d01e63b00bfec4ae"

    /** Written into every stored analysis so the review tool can tell the lanes apart. */
    const val BACKEND = "anime Chord-Transcription epoch150 · w30 o50"

    /** Seconds per feature frame, 512 / 22050. */
    const val SECONDS_PER_FRAME = HOP_LENGTH.toDouble() / SAMPLE_RATE.toDouble()
}
