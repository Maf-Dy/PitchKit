package com.nicos.pitchkit.tuner.harmony.btc

import android.content.Context
import com.nicos.pitchkit.tuner.AndroidModelAssets

object BtcAndroidFactory {
    fun songAssetsInstalled(context: Context): Boolean = BtcAssetFactory.songAssetsInstalled(AndroidModelAssets(context))

    fun liveAssetsInstalled(context: Context): Boolean = BtcAssetFactory.liveAssetsInstalled(AndroidModelAssets(context))

    fun createSongAnalyzer(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): BtcSongAnalyzer = BtcAssetFactory.createSongAnalyzer(AndroidModelAssets(context), referenceA4Hz, preferFlats)

    fun createLiveRecognizer(
        context: Context,
        referenceA4Hz: Double = 440.0,
    ): BtcStreamingRecognizer = BtcAssetFactory.createLiveRecognizer(AndroidModelAssets(context), referenceA4Hz)
}
