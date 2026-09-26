package com.nicos.pitchkit.tuner.harmony.animechord

import android.content.Context
import com.nicos.pitchkit.tuner.AndroidModelAssets

object AnimeChordAndroidFactory {
    fun songAssetsInstalled(context: Context): Boolean = AnimeChordAssetFactory.songAssetsInstalled(AndroidModelAssets(context))

    fun createSongAnalyzer(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = true,
    ): AnimeChordSongAnalyzer = AnimeChordAssetFactory.createSongAnalyzer(AndroidModelAssets(context), referenceA4Hz, preferFlats)
}
