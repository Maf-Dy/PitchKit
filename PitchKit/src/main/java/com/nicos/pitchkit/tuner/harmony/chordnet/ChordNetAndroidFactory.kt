package com.nicos.pitchkit.tuner.harmony.chordnet

import android.content.Context
import com.nicos.pitchkit.tuner.AndroidModelAssets

object ChordNetAndroidFactory {
    fun assetsInstalled(context: Context): Boolean = ChordNetAssetFactory.assetsInstalled(AndroidModelAssets(context))

    fun create(
        context: Context,
        referenceA4Hz: Double = 440.0,
    ): ChordNetStreamingRecognizer = ChordNetAssetFactory.create(AndroidModelAssets(context), referenceA4Hz)
}
