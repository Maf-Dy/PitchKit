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

    fun liveAssetsInstalled(context: Context): Boolean = songAssetsInstalled(context)

    fun createSongAnalyzer(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): BtcSongAnalyzer {
        val assets = loadAssets(context)
        return BtcSongAnalyzer(
            modelBytes = assets.model,
            metadata = assets.metadata,
            cqtPlanBytes = assets.plan,
            referenceA4Hz = referenceA4Hz,
            preferFlats = preferFlats,
        )
    }

    fun createLiveRecognizer(
        context: Context,
        referenceA4Hz: Double = 440.0,
    ): BtcStreamingRecognizer {
        val assets = loadAssets(context)
        return BtcStreamingRecognizer(
            modelBytes = assets.model,
            metadata = assets.metadata,
            cqtPlanBytes = assets.plan,
            referenceA4Hz = referenceA4Hz,
        )
    }

    private data class Assets(
        val model: ByteArray,
        val metadata: BtcMetadata,
        val plan: ByteArray,
    )

    private fun loadAssets(context: Context): Assets {
        require(songAssetsInstalled(context)) {
            "BTC assets are not installed. Run tools/fetch-btc.ps1 first."
        }
        val assets = context.assets
        val prefix = BtcContract.ASSET_DIR
        val model = assets.open("$prefix/${BtcContract.MODEL_FILE}").use { it.readBytes() }
        val metadataBytes = assets.open("$prefix/${BtcContract.METADATA_FILE}").use { it.readBytes() }
        val plan = assets.open("$prefix/${BtcContract.PLAN_FILE}").use { it.readBytes() }
        return Assets(
            model = model,
            metadata = BtcMetadata.parse(metadataBytes),
            plan = plan,
        )
    }
}
