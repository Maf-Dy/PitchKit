package com.nicos.pitchkit.tuner

import android.content.Context

class AndroidModelAssets(context: Context) : ModelAssets {
    private val assets = context.applicationContext.assets
    override fun list(directory: String) = assets.list(directory)
    override fun open(path: String) = assets.open(path)
}
