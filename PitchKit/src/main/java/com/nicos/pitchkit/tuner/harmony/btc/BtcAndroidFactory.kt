package com.nicos.pitchkit.tuner.harmony.btc

import android.content.Context

object BtcAndroidFactory {
    fun songAssetsInstalled(context: Context): Boolean {
        val assets = context.assets
        val names = runCatching { assets.list(BtcContract.ASSET_DIR)?.toSet().orEmpty() }
            .getOrDefault(emptySet())
        return BtcContract.MODEL_FILE in names &&
            BtcContract.METADATA_FILE in names &&
            BtcContract.PLAN_FILE in names
    }

    fun createSongAnalyzer(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): BtcSongAnalyzer {
        require(songAssetsInstalled(context)) {
            "BTC assets are not installed. Run tools/fetch-btc.ps1 first."
        }
        val assets = context.assets
        val prefix = BtcContract.ASSET_DIR
        val model = assets.open("$prefix/${BtcContract.MODEL_FILE}").use { it.readBytes() }
        val metadataBytes = assets.open("$prefix/${BtcContract.METADATA_FILE}").use { it.readBytes() }
        val plan = assets.open("$prefix/${BtcContract.PLAN_FILE}").use { it.readBytes() }
        return BtcSongAnalyzer(
            modelBytes = model,
            metadata = BtcMetadata.parse(metadataBytes),
            cqtPlanBytes = plan,
            referenceA4Hz = referenceA4Hz,
            preferFlats = preferFlats,
        )
    }
}
