package com.nicos.pitchkit.tuner.harmony.consonance

import com.nicos.pitchkit.tuner.ModelAssets

object ConsonanceAssetFactory {
    fun songAssetsInstalled(context: ModelAssets): Boolean {
        val names = runCatching {
            context.list(ConsonanceContract.ASSET_DIR)?.toSet().orEmpty()
        }.getOrDefault(emptySet())
        return ConsonanceContract.MODEL_FILE in names &&
            ConsonanceContract.METADATA_FILE in names &&
            ConsonanceContract.PLAN_FILE in names &&
            ConsonanceContract.DICTIONARY_FILE in names
    }

    /**
     * @param likelihood dictionary-decoder variant; null keeps the asset's own likelihood.
     * @param transitionPenalty chord-change penalty; null keeps the asset's own penalty.
     */
    fun createSongAnalyzer(
        context: ModelAssets,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
        likelihood: ConsonanceLikelihood? = null,
        transitionPenalty: Double? = null,
    ): ConsonanceSongAnalyzer {
        require(songAssetsInstalled(context)) {
            "Consonance assets are not installed. Run tools/fetch-consonance.ps1 first."
        }
        val prefix = ConsonanceContract.ASSET_DIR
        val assets = context
        val model = assets.open("$prefix/${ConsonanceContract.MODEL_FILE}").use { it.readBytes() }
        val metadata = assets.open("$prefix/${ConsonanceContract.METADATA_FILE}").use { it.readBytes() }
        val plan = assets.open("$prefix/${ConsonanceContract.PLAN_FILE}").use { it.readBytes() }
        val dictionary = assets.open("$prefix/${ConsonanceContract.DICTIONARY_FILE}")
            .use { it.readBytes() }
        return ConsonanceSongAnalyzer(
            modelBytes = model,
            metadata = ConsonanceMetadata.parse(metadata),
            cqtPlanBytes = plan,
            dictionaryBytes = dictionary,
            referenceA4Hz = referenceA4Hz,
            preferFlats = preferFlats,
            likelihood = likelihood,
            transitionPenalty = transitionPenalty,
        )
    }
}
