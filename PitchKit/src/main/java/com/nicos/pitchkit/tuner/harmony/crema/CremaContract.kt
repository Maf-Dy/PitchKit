package com.nicos.pitchkit.tuner.harmony.crema

object CremaContract {
    const val SAMPLE_RATE = 44_100
    const val HOP_LENGTH = 4_096
    const val INPUT_BINS = 216
    const val HARMONIC_CHANNELS = 2
    const val CHORD_COUNT = 170
    const val PITCH_COUNT = 12
    const val ROOT_COUNT = 13
    const val BASS_COUNT = 13

    const val INPUT_NAME = "cqt_mag"
    const val TAG_OUTPUT = "Identity:0"
    const val PITCH_OUTPUT = "Identity_1:0"
    const val ROOT_OUTPUT = "Identity_2:0"
    const val BASS_OUTPUT = "Identity_3:0"

    const val MODEL_FILE = "crema-0.2.0-opset18.onnx"
    const val STATE_FILE = "crema-0.2.0-runtime-state.json"

    const val MODEL_SIZE = 2_193_804
    const val STATE_SIZE = 3_790
    const val MODEL_SHA256 = "a903f9709821fccebb31d4e93d7d783642faaa90859f45f308c0f9131cc7ca59"
    const val STATE_SHA256 = "3744bf9ecb47de7194cb9f250fba26678ea347911af32ec4813645d5e033aca2"

    const val PLAN_H1_SHA256 = "65a1a732a8ed88a35e7641120745fc59e052ea843614817ffd71a44936dbf779"
    const val PLAN_H2_SHA256 = "8a44e4da0132d3bf3539aa3b5f88213e24a17d12c01964cac0b289fa3c758563"
}
