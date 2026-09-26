package com.nicos.pitchkit.tuner.harmony.solitito

import com.nicos.pitchkit.tuner.ModelAssets

/**
 * Loads the pinned solitito-ai artifacts out of the Android asset bundle.
 *
 * Two files: the 29 MB ONNX graph verbatim from upstream, and the packed DSP plan that
 * `tools/accuracy_audit/export_solitito_assets.py` writes from upstream's 2 MB
 * `dsp_weights.json`. Both are SHA-256-guarded inside [SolititoStreamingRecognizer], so a
 * swapped or truncated asset fails the construction rather than quietly changing the
 * front end.
 *
 * There is no `referenceA4Hz` here, unlike the other factories. solitito's kernel is a
 * fixed 144-bin pseudo-CQT from C1 baked into the shipped weights, and its root head is a
 * 13-way classifier over pitch-class names rather than a frequency read-out, so there is
 * nothing a reference pitch could retune without regenerating the asset.
 */
object SolititoAssetFactory {
    fun assetsInstalled(context: ModelAssets): Boolean {
        val names = context.list(SolititoContract.ASSET_DIRECTORY)?.toSet().orEmpty()
        return SolititoContract.MODEL_FILE in names && SolititoContract.PLAN_FILE in names
    }

    fun create(context: ModelAssets): SolititoStreamingRecognizer {
        require(assetsInstalled(context)) {
            "solitito assets are not installed. Run " +
                "tools/accuracy_audit/export_solitito_assets.py first."
        }
        val model = context
            .open("${SolititoContract.ASSET_DIRECTORY}/${SolititoContract.MODEL_FILE}")
            .use { it.readBytes() }
        val plan = context
            .open("${SolititoContract.ASSET_DIRECTORY}/${SolititoContract.PLAN_FILE}")
            .use { it.readBytes() }
        return SolititoStreamingRecognizer(modelBytes = model, planBytes = plan)
    }
}
