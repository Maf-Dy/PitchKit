package com.nicos.pitchkit.tuner.harmony.chordformer

import com.nicos.pitchkit.tuner.ModelAssets
import org.json.JSONObject
import java.security.MessageDigest

internal data class ChordFormerAssetSpec(
    val file: String,
    val size: Int,
    val sha256: String,
)

/** Offline-only ChordFormer asset factory. */
object ChordFormerAssetFactory {
    fun songAssetsInstalled(context: ModelAssets): Boolean {
        val names = context.list(ChordFormerContract.ASSET_DIRECTORY)?.toSet().orEmpty()
        return ChordFormerContract.METADATA_FILE in names &&
            ChordFormerContract.MODEL_FILE in names &&
            ChordFormerContract.DICTIONARY_FILE in names &&
            ChordFormerContract.LOW_CQT_PLAN_FILE in names &&
            ChordFormerContract.HIGH_CQT_PLAN_FILE in names
    }

    fun createSongAnalyzer(
        context: ModelAssets,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): ChordFormerSongAnalyzer {
        require(songAssetsInstalled(context)) {
            "ChordFormer song assets are not installed. " +
                "Run tools/generate-chordformer-cqt-plans.py and " +
                "tools/accuracy_audit/export_chordformer.py first."
        }
        val metadata = readMetadata(context)

        return ChordFormerSongAnalyzer(
            modelBytes = readVerified(context, metadata.model, ChordFormerContract.MODEL_SHA256),
            dictionaryJson = readVerified(
                context,
                metadata.dictionary,
                ChordFormerContract.DICTIONARY_SHA256,
            ).toString(Charsets.UTF_8),
            lowPlanBytes = readVerified(context, metadata.lowPlan),
            highPlanBytes = readVerified(context, metadata.highPlan),
            referenceA4Hz = referenceA4Hz,
            preferFlats = preferFlats,
        )
    }

    private class Metadata(
        val model: ChordFormerAssetSpec,
        val dictionary: ChordFormerAssetSpec,
        val lowPlan: ChordFormerAssetSpec,
        val highPlan: ChordFormerAssetSpec,
    )

    private fun readMetadata(context: ModelAssets): Metadata {
        val bytes = readBytes(context, ChordFormerContract.METADATA_FILE)
        val root = JSONObject(bytes.toString(Charsets.UTF_8))
        require(root.getInt("schema_version") == 1)
        require(root.getString("backend") == ChordFormerContract.BACKEND) {
            "Unexpected ChordFormer backend: ${root.getString("backend")}"
        }
        require(root.getString("source_commit") == ChordFormerContract.SOURCE_COMMIT)
        require(root.getString("checkpoint") == ChordFormerContract.CHECKPOINT)
        require(root.getString("checkpoint_sha256") == ChordFormerContract.CHECKPOINT_SHA256)
        require(root.getInt("window_frames") == ChordFormerContract.WINDOW_FRAMES)
        require(root.getInt("overlap_frames") == ChordFormerContract.OVERLAP_FRAMES)
        require(root.getBoolean("fixed_sequence_length"))
        require(root.getString("input_name") == ChordFormerContract.INPUT_NAME)

        val cqt = root.getJSONObject("cqt")
        require(cqt.getInt("sample_rate") == ChordFormerContract.SAMPLE_RATE)
        require(cqt.getInt("hop_length") == ChordFormerContract.HOP_LENGTH)
        require(cqt.getInt("bins_per_octave") == ChordFormerContract.BINS_PER_OCTAVE)
        require(cqt.getInt("original_bins") == ChordFormerContract.ORIGINAL_HYBRID_BINS)
        require(cqt.getInt("input_bins") == ChordFormerContract.INPUT_BINS)
        val crop = cqt.getJSONArray("model_crop")
        require(crop.getInt(0) == ChordFormerContract.DROP_LOW_BINS)
        require(crop.getInt(1) == ChordFormerContract.DROP_LOW_BINS + ChordFormerContract.INPUT_BINS)

        val decoder = root.getJSONObject("decoder")
        require(decoder.getString("vocabulary") == ChordFormerContract.VOCABULARY)
        require(decoder.getDouble("transition_penalty") == ChordFormerContract.TRANSITION_PENALTY)
        require(!decoder.getBoolean("beat_aware"))

        val plans = cqt.getJSONArray("plans")
        var low: ChordFormerAssetSpec? = null
        var high: ChordFormerAssetSpec? = null
        for (index in 0 until plans.length()) {
            val spec = parseSpec(plans.getJSONObject(index))
            when (spec.file) {
                ChordFormerContract.LOW_CQT_PLAN_FILE -> low = spec
                ChordFormerContract.HIGH_CQT_PLAN_FILE -> high = spec
            }
        }

        return Metadata(
            model = ChordFormerAssetSpec(
                file = root.getString("model_file"),
                size = root.getInt("model_size"),
                sha256 = root.getString("model_sha256"),
            ),
            dictionary = ChordFormerAssetSpec(
                file = decoder.getString("dictionary_file"),
                size = decoder.getInt("dictionary_size"),
                sha256 = decoder.getString("dictionary_sha256"),
            ),
            lowPlan = requireNotNull(low) { "ChordFormer metadata is missing the recursive CQT plan" },
            highPlan = requireNotNull(high) { "ChordFormer metadata is missing the pseudo-CQT plan" },
        )
    }

    private fun parseSpec(json: JSONObject): ChordFormerAssetSpec = ChordFormerAssetSpec(
        file = json.getString("file"),
        size = json.getInt("size"),
        sha256 = json.getString("sha256"),
    )

    /**
     * @param pinned an expected digest compiled into the app, so a swapped metadata file
     * cannot quietly authorise a different model or dictionary.
     */
    private fun readVerified(
        context: ModelAssets,
        spec: ChordFormerAssetSpec,
        pinned: String? = null,
    ): ByteArray {
        require(pinned == null || pinned == spec.sha256) {
            "ChordFormer metadata declares an unexpected digest for ${spec.file}"
        }
        val bytes = readBytes(context, spec.file)
        val hash = sha256(bytes)
        require(bytes.size == spec.size && hash == spec.sha256) {
            "Unexpected ChordFormer asset ${spec.file}: size=${bytes.size}, sha256=$hash"
        }
        return bytes
    }

    private fun readBytes(context: ModelAssets, file: String): ByteArray = context
        .open("${ChordFormerContract.ASSET_DIRECTORY}/$file")
        .use { it.readBytes() }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
