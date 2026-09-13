package com.nicos.pitchkit.tuner.harmony.lvchordia

internal object LvChordiaContract {
    const val ASSET_DIRECTORY = "lvchordia"
    const val MANIFEST_FILE = "manifest.json"
    const val DICTIONARY_FILE = "full-dictionary.json"
    const val LOW_CQT_PLAN_FILE = "lvchordia-cqt-low.bin"
    const val HIGH_CQT_PLAN_FILE = "lvchordia-cqt-high.bin"

    const val SAMPLE_RATE = 22_050
    const val HOP_LENGTH = 512
    const val INPUT_BINS = 252

    const val ORIGINAL_HYBRID_BINS = 288
    const val ORIGINAL_RECURSIVE_BINS = 238
    const val DROP_LOW_BINS = 18
    const val RECURSIVE_MODEL_BINS = ORIGINAL_RECURSIVE_BINS - DROP_LOW_BINS
    const val PSEUDO_MODEL_BINS = INPUT_BINS - RECURSIVE_MODEL_BINS

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

    val MODEL_FILES = List(5) { index -> "lvchordia-s$index.onnx" }

    const val SOURCE_PACKAGE_VERSION = "1.1.0"
}
