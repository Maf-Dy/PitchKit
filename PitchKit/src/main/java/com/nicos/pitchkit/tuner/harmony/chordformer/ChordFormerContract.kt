package com.nicos.pitchkit.tuner.harmony.chordformer

/**
 * Asset and graph contract for the offline ChordFormer song engine.
 *
 * ChordFormer (github.com/mwaseemrandhawa/ChordFormer, pinned commit 3da11c0) is a
 * four-layer conformer trained on the LV-Chordia data with the same six chord heads
 * (73/13/4/4/3/3 = 100 columns) and the same dictionary XHMM decoder LV Song uses, so
 * the decoding half of this engine is literally LV Song's. Two things differ:
 *
 *  * the feature is CQTV2 *with* `librosa.amplitude_to_db(..., ref=np.max)` applied over
 *    the whole 288-bin hybrid spectrogram before the 18..270 model crop, where LV Song
 *    feeds raw magnitudes;
 *  * `nn.MultiheadAttention` bakes the traced sequence length into a Reshape, so the
 *    exported graph only runs at one fixed window length.
 *
 * Produced by tools/accuracy_audit/export_chordformer.py and
 * tools/generate-chordformer-cqt-plans.py.
 */
internal object ChordFormerContract {
    const val ASSET_DIRECTORY = "chordformer"
    const val METADATA_FILE = "chordformer-meta.json"
    const val MODEL_FILE = "chordformer-fold3-t2048.onnx"
    const val DICTIONARY_FILE = "chordformer-dictionary.json"
    const val LOW_CQT_PLAN_FILE = "chordformer-cqt-low.bin"
    const val HIGH_CQT_PLAN_FILE = "chordformer-cqt-high.bin"

    const val SAMPLE_RATE = 22_050
    const val HOP_LENGTH = 512
    const val BINS_PER_OCTAVE = 36
    const val INPUT_BINS = 252

    const val ORIGINAL_HYBRID_BINS = 288
    const val RECURSIVE_BINS = 238
    const val PSEUDO_BINS = 50
    const val DROP_LOW_BINS = 18

    /** librosa `amplitude_to_db` constants: amin=1e-5 on amplitudes, top_db=80. */
    const val AMPLITUDE_FLOOR = 1e-5
    const val TOP_DB = 80.0

    /**
     * The reference is `ref=np.max`, so the loudest bin of the song is 0 dB and the
     * floor is exactly -80 dB. True silence lands on the floor, which is therefore also
     * the honest pad value for a short or trailing window.
     */
    const val SILENCE_DB = -80.0f

    /**
     * Fixed exported sequence length. The feasibility run measured 2048/512 windows at
     * only -0.7 root / -1.7 family against the unwindowed full-song reference.
     */
    const val WINDOW_FRAMES = 2048
    const val OVERLAP_FRAMES = 512
    const val HALF_OVERLAP_FRAMES = OVERLAP_FRAMES / 2

    const val TRIAD_COUNT = 73
    const val BASS_COUNT = 13
    const val SEVENTH_COUNT = 4
    const val NINTH_COUNT = 4
    const val ELEVENTH_COUNT = 3
    const val THIRTEENTH_COUNT = 3

    const val INPUT_NAME = "cqt"
    const val TRIAD_OUTPUT = "triad"
    const val BASS_OUTPUT = "bass"
    const val SEVENTH_OUTPUT = "seventh"
    const val NINTH_OUTPUT = "ninth"
    const val ELEVENTH_OUTPUT = "eleventh"
    const val THIRTEENTH_OUTPUT = "thirteenth"

    const val SOURCE_COMMIT = "3da11c078c802b3f5a2ff0f6f8ec184a41537844"
    const val CHECKPOINT = "cache_data/chordformer_head16(1.0,1.0)_s3.best.sdict"
    const val CHECKPOINT_SHA256 =
        "0e04b933293ad7cb6895c91a63aa6940832adb46a1b895e77afd3d837a24af1b"
    const val MODEL_SHA256 =
        "93aee1aec282a10a2177cb51b1f3b871b0c7704f5b73c0d25180f8263511bff0"
    const val DICTIONARY_SHA256 =
        "ef31a56d67bbafc41c963ff9e4fbf4b0bcb8ea22a51c05183f33de459d801fca"

    const val VOCABULARY = "submission"
    const val TRANSITION_PENALTY = 30.0

    /**
     * Written into every stored analysis so the review tool can tell the lanes apart.
     * `unprimed` marks the window plan that matches the PC reference lane: window 0 starts
     * at real frame 0 with no reflected lead-in (see
     * `.accuracy-work/annotations/chordformer-phone-diagnosis.md`).
     */
    const val BACKEND = "ChordFormer fold3 · submission · t2048 · unprimed"
}
