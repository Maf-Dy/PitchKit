package com.nicos.pitchkit.tuner.harmony.chordnet

import android.content.Context

/** Loads the pinned model artifacts from the Android asset bundle. */
object ChordNetAndroidFactory {
    private const val ASSET_DIRECTORY = "chordnet"

    fun assetsInstalled(context: Context): Boolean {
        val names = context.assets.list(ASSET_DIRECTORY)?.toSet().orEmpty()
        return ChordNetContract.MODEL_FILE in names && ChordNetContract.PLAN_FILE in names
    }

    fun create(
        context: Context,
        referenceA4Hz: Double = 440.0,
    ): ChordNetStreamingRecognizer {
        require(assetsInstalled(context)) {
            "ChordNet assets are not installed. Run tools/fetch-chordnet.ps1 first."
        }

        val model = context.assets
            .open("$ASSET_DIRECTORY/${ChordNetContract.MODEL_FILE}")
            .use { it.readBytes() }
        val plan = context.assets
            .open("$ASSET_DIRECTORY/${ChordNetContract.PLAN_FILE}")
            .use { it.readBytes() }

        return ChordNetStreamingRecognizer(
            modelBytes = model,
            planBytes = plan,
            referenceA4Hz = referenceA4Hz,
        )
    }
}
