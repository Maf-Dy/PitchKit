package com.nicos.pitchkit.tuner.harmony.crema

import android.content.Context
import com.nicos.pitchkit.tuner.AndroidModelAssets

object CremaAndroidFactory {
    fun assetsInstalled(context: Context): Boolean = CremaAssetFactory.assetsInstalled(AndroidModelAssets(context))

    fun songAssetsInstalled(context: Context): Boolean = CremaAssetFactory.songAssetsInstalled(AndroidModelAssets(context))

    fun create(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): CremaStreamingRecognizer = CremaAssetFactory.create(AndroidModelAssets(context), referenceA4Hz, preferFlats)

    fun createSongAnalyzer(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): CremaSongAnalyzer = CremaAssetFactory.createSongAnalyzer(AndroidModelAssets(context), referenceA4Hz, preferFlats)
}
