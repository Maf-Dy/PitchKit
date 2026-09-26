package com.nicos.pitchkit.tuner.harmony.animechord

import com.nicos.pitchkit.tuner.ModelAssets
import org.json.JSONObject
import java.security.MessageDigest

internal data class AnimeChordAssetSpec(
    val file: String,
    val size: Int,
    val sha256: String,
)

/** Offline-only `anime Chord-Transcription` asset factory. */
object AnimeChordAssetFactory {
    fun songAssetsInstalled(context: ModelAssets): Boolean {
        val names = context.list(AnimeChordContract.ASSET_DIRECTORY)?.toSet().orEmpty()
        return AnimeChordContract.METADATA_FILE in names &&
            AnimeChordContract.MODEL_FILE in names &&
            AnimeChordContract.CQT_PLAN_FILE in names &&
            AnimeChordContract.VOCABULARY_FILE in names
    }

    fun createSongAnalyzer(
        context: ModelAssets,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = true,
    ): AnimeChordSongAnalyzer {
        require(songAssetsInstalled(context)) {
            "anime Chord-Transcription song assets are not installed. " +
                "Run tools/generate-animechord-cqt-plans.py first."
        }
        val metadata = readMetadata(context)
        return AnimeChordSongAnalyzer(
            modelBytes = readVerified(context, metadata.model, AnimeChordContract.MODEL_SHA256),
            planBytes = readVerified(context, metadata.plan, AnimeChordContract.CQT_PLAN_SHA256),
            vocabularyJson = readVerified(
                context,
                metadata.vocabulary,
                AnimeChordContract.VOCABULARY_SHA256,
            ).toString(Charsets.UTF_8),
            referenceA4Hz = referenceA4Hz,
            preferFlats = preferFlats,
        )
    }

    private class Metadata(
        val model: AnimeChordAssetSpec,
        val plan: AnimeChordAssetSpec,
        val vocabulary: AnimeChordAssetSpec,
    )

    private fun readMetadata(context: ModelAssets): Metadata {
        val root = JSONObject(readBytes(context, AnimeChordContract.METADATA_FILE)
            .toString(Charsets.UTF_8))
        require(root.getInt("schema_version") == 1)
        require(root.getString("backend") == AnimeChordContract.BACKEND) {
            "Unexpected anime backend: ${root.getString("backend")}"
        }
        require(root.getString("checkpoint") == AnimeChordContract.CHECKPOINT)
        require(root.getString("checkpoint_sha256") == AnimeChordContract.CHECKPOINT_SHA256)
        require(root.getString("hf_revision") == AnimeChordContract.HF_REVISION)
        require(root.getString("source_commit") == AnimeChordContract.SOURCE_COMMIT)
        require(root.getString("input_name") == AnimeChordContract.INPUT_NAME)
        require(root.getBoolean("dynamic_time_axis"))
        require(root.getInt("window_frames") == AnimeChordContract.WINDOW_FRAMES)
        require(root.getInt("overlap_frames") == AnimeChordContract.OVERLAP_FRAMES)
        require(!root.getBoolean("first_window_primed"))

        val cqt = root.getJSONObject("cqt")
        require(cqt.getInt("sample_rate") == AnimeChordContract.SAMPLE_RATE)
        require(cqt.getInt("hop_length") == AnimeChordContract.HOP_LENGTH)
        require(cqt.getInt("n_bins") == AnimeChordContract.INPUT_BINS)
        require(cqt.getInt("bins_per_octave") == AnimeChordContract.BINS_PER_OCTAVE)
        require(cqt.getInt("octaves") == AnimeChordContract.OCTAVES)
        require(cqt.getInt("stage_n_fft") == AnimeChordContract.STAGE_FFT)
        require(cqt.getInt("decimator_taps") == AnimeChordContract.DECIMATOR_TAPS)
        require(cqt.getInt("crop_n_fft") == AnimeChordContract.CROP_N_FFT)
        require(cqt.getDouble("filter_scale") == AnimeChordContract.FILTER_SCALE)

        val decoder = root.getJSONObject("decoder")
        require(decoder.getString("kind") == "sticky-hmm-viterbi")
        require(decoder.getDouble("root_chord_stay_prob") == 1.0)
        require(decoder.getDouble("bass_stay_prob") == 1.0)
        require(decoder.getDouble("log_stay") == AnimeChordContract.LOG_STAY)
        require(decoder.getDouble("min_duration_chord") == AnimeChordContract.MIN_CHORD_SECONDS)

        val vocabulary = root.getJSONObject("vocabulary")
        require(vocabulary.getInt("root_classes") == AnimeChordContract.ROOT_CHORD_COUNT)
        require(vocabulary.getInt("bass_classes") == AnimeChordContract.BASS_COUNT)
        require(vocabulary.getInt("quality_classes") == AnimeChordContract.QUALITY_COUNT)

        return Metadata(
            model = AnimeChordAssetSpec(
                file = root.getString("model_file"),
                size = root.getInt("model_size"),
                sha256 = root.getString("model_sha256"),
            ),
            plan = AnimeChordAssetSpec(
                file = cqt.getString("file"),
                size = cqt.getInt("size"),
                sha256 = cqt.getString("sha256"),
            ),
            vocabulary = AnimeChordAssetSpec(
                file = vocabulary.getString("file"),
                size = vocabulary.getInt("size"),
                sha256 = vocabulary.getString("sha256"),
            ),
        )
    }

    /**
     * @param pinned an expected digest compiled into the app, so a swapped metadata file
     * cannot quietly authorise a different model, front end or vocabulary.
     */
    private fun readVerified(
        context: ModelAssets,
        spec: AnimeChordAssetSpec,
        pinned: String,
    ): ByteArray {
        require(pinned == spec.sha256) {
            "anime metadata declares an unexpected digest for ${spec.file}"
        }
        val bytes = readBytes(context, spec.file)
        val hash = sha256(bytes)
        require(bytes.size == spec.size && hash == spec.sha256) {
            "Unexpected anime asset ${spec.file}: size=${bytes.size}, sha256=$hash"
        }
        return bytes
    }

    private fun readBytes(context: ModelAssets, file: String): ByteArray = context
        .open("${AnimeChordContract.ASSET_DIRECTORY}/$file")
        .use { it.readBytes() }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
