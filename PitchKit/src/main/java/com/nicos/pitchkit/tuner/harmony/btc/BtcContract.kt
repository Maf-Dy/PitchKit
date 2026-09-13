package com.nicos.pitchkit.tuner.harmony.btc

object BtcContract {
    const val SAMPLE_RATE = 22_050
    const val HOP_LENGTH = 2_048
    const val INPUT_BINS = 144
    const val SEQUENCE_LENGTH = 108
    const val CHORD_COUNT = 170

    const val WINDOW_STRIDE = 54
    const val SMOOTHING_KERNEL = 9
    const val MIN_SEGMENT_SECONDS = 0.5

    const val INPUT_NAME = "features"
    const val OUTPUT_NAME = "logits"

    const val ASSET_DIR = "btc"
    const val MODEL_FILE = "btc.onnx"
    const val METADATA_FILE = "btc-meta.properties"
    const val PLAN_FILE = "cqt-plan.bin"

    // BTC uses the same librosa-compatible CQT plan as our ChordNet path.
    const val PLAN_SHA256 = "c31f0a6fd2d582d753be6628b5daecdee58acba53cba93b2bc2b5c75dee2ba48"
}
