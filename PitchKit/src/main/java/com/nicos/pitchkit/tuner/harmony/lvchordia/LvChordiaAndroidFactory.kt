package com.nicos.pitchkit.tuner.harmony.lvchordia

import android.content.Context
import com.nicos.pitchkit.tuner.AndroidModelAssets

object LvChordiaAndroidFactory {
    fun songAssetsInstalled(context: Context): Boolean = LvChordiaAssetFactory.songAssetsInstalled(AndroidModelAssets(context))

    fun createSongAnalyzer(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): LvChordiaSongAnalyzer = LvChordiaAssetFactory.createSongAnalyzer(AndroidModelAssets(context), referenceA4Hz, preferFlats)
}
