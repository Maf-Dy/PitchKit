package com.nicos.pitchkit.tuner.harmony.consonance

object ConsonanceContract {
    const val SAMPLE_RATE = 22_050
    const val HOP_LENGTH = 512
    const val INPUT_BINS = 144
    const val BINS_PER_OCTAVE = 24
    const val ROOT_COUNT = 13
    const val BASS_COUNT = 13
    const val PITCH_COUNT = 12

    const val CHUNK_SECONDS = 20
    const val PITCH_THRESHOLD = 0.50f
    const val MIN_SEGMENT_SECONDS = 0.50

    const val INPUT_NAME = "cqt"
    const val ROOT_OUTPUT = "root"
    const val BASS_OUTPUT = "bass"
    const val CHORD_OUTPUT = "chord"

    const val ASSET_DIR = "consonance"
    const val MODEL_FILE = "consonance-decomposed.onnx"
    const val METADATA_FILE = "consonance-meta.json"
    const val PLAN_FILE = "consonance-cqt-plan.bin"

    const val SOURCE_COMMIT = "d17633aea4e68e616e735d09df97b97ae3428e71"
    const val PLAN_SHA256 = "419bf3d2aa82dc58b56670620e8351be5806db7e49730b1ef657e9e806f1bab0"
}
