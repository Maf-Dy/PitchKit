package com.nicos.pitchkit.tuner.harmony.chordnet

object ChordNetContract {
    const val SAMPLE_RATE = 22_050
    const val HOP_LENGTH = 2_048
    const val INPUT_BINS = 144
    const val SEQUENCE_LENGTH = 108
    const val CHORD_COUNT = 170
    const val SMOOTHING_KERNEL = 9

    const val INPUT_NAME = "features"
    const val OUTPUT_NAME = "logits"

    const val MODEL_FILE = "chordnet.onnx"
    const val PLAN_FILE = "cqt-plan.bin"

    const val MODEL_SHA256 = "9a6570bf611cdc3f2c36286307af46fb94927fe7f6a2bc22a87c0ebf5f6c082e"
    const val PLAN_SHA256 = "c31f0a6fd2d582d753be6628b5daecdee58acba53cba93b2bc2b5c75dee2ba48"
}
