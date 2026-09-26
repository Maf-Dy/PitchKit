package com.nicos.pitchkit.tuner.harmony.chordformer

import android.content.Context
import com.nicos.pitchkit.tuner.AndroidModelAssets

object ChordFormerAndroidFactory {
    fun songAssetsInstalled(context: Context): Boolean = ChordFormerAssetFactory.songAssetsInstalled(AndroidModelAssets(context))

    fun createSongAnalyzer(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): ChordFormerSongAnalyzer = ChordFormerAssetFactory.createSongAnalyzer(AndroidModelAssets(context), referenceA4Hz, preferFlats)
}
