package com.nicos.pitchkit.tuner.harmony.consonance

import android.content.Context

object ConsonanceAndroidFactory {
    fun songAssetsInstalled(context: Context): Boolean {
        val names = runCatching {
            context.assets.list(ConsonanceContract.ASSET_DIR)?.toSet().orEmpty()
        }.getOrDefault(emptySet())
        return ConsonanceContract.MODEL_FILE in names &&
            ConsonanceContract.METADATA_FILE in names &&
            ConsonanceContract.PLAN_FILE in names
    }

    fun createSongAnalyzer(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): ConsonanceSongAnalyzer {
        require(songAssetsInstalled(context)) {
            "Consonance assets are not installed. Run tools/fetch-consonance.ps1 first."
        }
        val prefix = ConsonanceContract.ASSET_DIR
        val assets = context.assets
        val model = assets.open("$prefix/${ConsonanceContract.MODEL_FILE}").use { it.readBytes() }
        val metadata = assets.open("$prefix/${ConsonanceContract.METADATA_FILE}").use { it.readBytes() }
        val plan = assets.open("$prefix/${ConsonanceContract.PLAN_FILE}").use { it.readBytes() }
        return ConsonanceSongAnalyzer(
            modelBytes = model,
            metadata = ConsonanceMetadata.parse(metadata),
            cqtPlanBytes = plan,
            referenceA4Hz = referenceA4Hz,
            preferFlats = preferFlats,
        )
    }
}
