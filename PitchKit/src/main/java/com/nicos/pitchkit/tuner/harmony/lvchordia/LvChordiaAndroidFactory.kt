package com.nicos.pitchkit.tuner.harmony.lvchordia

import android.content.Context
import org.json.JSONObject
import java.security.MessageDigest

internal data class LvChordiaAssetSpec(
    val file: String,
    val size: Int,
    val sha256: String,
)

internal data class LvChordiaManifest(
    val models: List<LvChordiaAssetSpec>,
    val dictionary: LvChordiaAssetSpec,
    val frontend: List<LvChordiaAssetSpec>,
)

/** Offline-only LV-Chordia asset factory. */
object LvChordiaAndroidFactory {
    fun songAssetsInstalled(context: Context): Boolean {
        val names = context.assets.list(LvChordiaContract.ASSET_DIRECTORY)?.toSet().orEmpty()
        return LvChordiaContract.MANIFEST_FILE in names &&
            LvChordiaContract.DICTIONARY_FILE in names &&
            LvChordiaContract.LOW_CQT_PLAN_FILE in names &&
            LvChordiaContract.HIGH_CQT_PLAN_FILE in names &&
            LvChordiaContract.MODEL_FILES.all { it in names }
    }

    fun createSongAnalyzer(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): LvChordiaSongAnalyzer {
        require(songAssetsInstalled(context)) {
            "LV-Chordia song assets are not installed. Run tools/fetch-lvchordia.ps1 first."
        }
        val manifest = readManifest(context)
        val modelBytes = LvChordiaContract.MODEL_FILES.map { file ->
            val spec = manifest.models.firstOrNull { it.file == file }
                ?: error("LV-Chordia manifest does not contain $file")
            readVerified(context, spec)
        }
        val lowSpec = manifest.frontend.firstOrNull {
            it.file == LvChordiaContract.LOW_CQT_PLAN_FILE
        } ?: error("LV-Chordia manifest does not contain the recursive CQT plan")
        val highSpec = manifest.frontend.firstOrNull {
            it.file == LvChordiaContract.HIGH_CQT_PLAN_FILE
        } ?: error("LV-Chordia manifest does not contain the pseudo-CQT plan")

        return LvChordiaSongAnalyzer(
            modelBytes = modelBytes,
            dictionaryJson = readVerified(context, manifest.dictionary).toString(Charsets.UTF_8),
            lowPlanBytes = readVerified(context, lowSpec),
            highPlanBytes = readVerified(context, highSpec),
            referenceA4Hz = referenceA4Hz,
            preferFlats = preferFlats,
        )
    }

    private fun readManifest(context: Context): LvChordiaManifest {
        val bytes = readBytes(context, LvChordiaContract.MANIFEST_FILE)
        val root = JSONObject(bytes.toString(Charsets.UTF_8))
        require(root.getInt("schema_version") == 1)
        require(root.getString("source_package_version") == LvChordiaContract.SOURCE_PACKAGE_VERSION) {
            "Unexpected LV-Chordia package version: ${root.getString("source_package_version")}"
        }
        require(root.getInt("sample_rate") == LvChordiaContract.SAMPLE_RATE)
        require(root.getInt("hop_length") == LvChordiaContract.HOP_LENGTH)
        require(root.getInt("input_bins") == LvChordiaContract.INPUT_BINS)

        val modelsJson = root.getJSONArray("models")
        val models = List(modelsJson.length()) { index -> parseSpec(modelsJson.getJSONObject(index)) }
        require(models.size == LvChordiaContract.MODEL_FILES.size)

        val dictionary = parseSpec(root.getJSONObject("dictionary"))
        val frontendRoot = root.getJSONObject("frontend")
        require(frontendRoot.getString("kind") == "librosa-hybrid-cqt")
        require(frontendRoot.getInt("sample_rate") == LvChordiaContract.SAMPLE_RATE)
        require(frontendRoot.getInt("hop_length") == LvChordiaContract.HOP_LENGTH)
        require(frontendRoot.getInt("original_bins") == LvChordiaContract.ORIGINAL_HYBRID_BINS)
        require(frontendRoot.getInt("input_bins") == LvChordiaContract.INPUT_BINS)
        require(frontendRoot.getInt("recursive_bins") == LvChordiaContract.ORIGINAL_RECURSIVE_BINS)
        require(frontendRoot.getInt("pseudo_bins") == 50)
        val filesJson = frontendRoot.getJSONArray("files")
        val frontend = List(filesJson.length()) { index -> parseSpec(filesJson.getJSONObject(index)) }

        return LvChordiaManifest(models, dictionary, frontend)
    }

    private fun parseSpec(json: JSONObject): LvChordiaAssetSpec = LvChordiaAssetSpec(
        file = json.getString("file"),
        size = json.getInt("size"),
        sha256 = json.getString("sha256"),
    )

    private fun readVerified(context: Context, spec: LvChordiaAssetSpec): ByteArray {
        val bytes = readBytes(context, spec.file)
        val hash = sha256(bytes)
        require(bytes.size == spec.size && hash == spec.sha256) {
            "Unexpected LV-Chordia asset ${spec.file}: size=${bytes.size}, sha256=$hash"
        }
        return bytes
    }

    private fun readBytes(context: Context, file: String): ByteArray = context.assets
        .open("${LvChordiaContract.ASSET_DIRECTORY}/$file")
        .use { it.readBytes() }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
