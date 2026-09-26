package com.nicos.pitchkit.tuner.harmony.chordnet

import com.nicos.pitchkit.tuner.ModelAssets

/** Loads the pinned model artifacts from the Android asset bundle. */
object ChordNetAssetFactory {
    private const val ASSET_DIRECTORY = "chordnet"

    fun assetsInstalled(context: ModelAssets): Boolean {
        val names = context.list(ASSET_DIRECTORY)?.toSet().orEmpty()
        return ChordNetContract.MODEL_FILE in names && ChordNetContract.PLAN_FILE in names
    }

    fun create(
        context: ModelAssets,
        referenceA4Hz: Double = 440.0,
        stageTiming: ((String, Long) -> Unit)? = null,
    ): ChordNetStreamingRecognizer {
        require(assetsInstalled(context)) {
            "ChordNet assets are not installed. Run tools/fetch-chordnet.ps1 first."
        }

        val model = context
            .open("$ASSET_DIRECTORY/${ChordNetContract.MODEL_FILE}")
            .use { it.readBytes() }
        val plan = context
            .open("$ASSET_DIRECTORY/${ChordNetContract.PLAN_FILE}")
            .use { it.readBytes() }

        return ChordNetStreamingRecognizer(
            modelBytes = model,
            planBytes = plan,
            referenceA4Hz = referenceA4Hz,
            stageTiming = stageTiming,
        )
    }
}
