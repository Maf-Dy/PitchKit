package com.nicos.pitchkit.tuner.harmony.consonance

import android.content.Context
import com.nicos.pitchkit.tuner.AndroidModelAssets

object ConsonanceAndroidFactory {
    fun songAssetsInstalled(context: Context): Boolean = ConsonanceAssetFactory.songAssetsInstalled(AndroidModelAssets(context))

    fun createSongAnalyzer(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
        likelihood: ConsonanceLikelihood? = null,
        transitionPenalty: Double? = null,
    ): ConsonanceSongAnalyzer = ConsonanceAssetFactory.createSongAnalyzer(AndroidModelAssets(context), referenceA4Hz, preferFlats, likelihood, transitionPenalty)
}
