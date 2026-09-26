package com.nicos.pitchkit.tuner.harmony.solitito

import android.content.Context
import com.nicos.pitchkit.tuner.AndroidModelAssets

object SolititoAndroidFactory {
    fun assetsInstalled(context: Context): Boolean = SolititoAssetFactory.assetsInstalled(AndroidModelAssets(context))

    fun create(context: Context): SolititoStreamingRecognizer = SolititoAssetFactory.create(AndroidModelAssets(context))
}
