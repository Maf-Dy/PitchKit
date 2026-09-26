package com.nicos.pitchkit.tuner.harmony.chordformer

import java.security.MessageDigest
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The shipped assets, the metadata they carry and the constants the app compiles against
 * all have to name the same fold-3 build. The Android factory checks this at runtime;
 * this catches a mismatched asset drop before it ever reaches a device.
 */
class ChordFormerContractTest {
    @Test
    fun metadataAgreesWithTheCompiledContract() {
        val metadata = JSONObject(
            ChordFormerTestAssets.locate(ChordFormerContract.METADATA_FILE)
                .readBytes()
                .toString(Charsets.UTF_8)
        )
        assertEquals(1, metadata.getInt("schema_version"))
        assertEquals(ChordFormerContract.BACKEND, metadata.getString("backend"))
        assertEquals(ChordFormerContract.SOURCE_COMMIT, metadata.getString("source_commit"))
        assertEquals(ChordFormerContract.CHECKPOINT, metadata.getString("checkpoint"))
        assertEquals(
            ChordFormerContract.CHECKPOINT_SHA256,
            metadata.getString("checkpoint_sha256"),
        )
        assertEquals(ChordFormerContract.MODEL_FILE, metadata.getString("model_file"))
        assertEquals(ChordFormerContract.MODEL_SHA256, metadata.getString("model_sha256"))
        assertEquals(ChordFormerContract.WINDOW_FRAMES, metadata.getInt("window_frames"))
        assertEquals(ChordFormerContract.OVERLAP_FRAMES, metadata.getInt("overlap_frames"))
        assertEquals(17, metadata.getInt("onnx_opset"))
        assertTrue(metadata.getBoolean("fixed_sequence_length"))
        assertTrue(metadata.getBoolean("graph_reexport_reproduces_archive"))

        // The feasibility run's parity budget for the fixed-length graph.
        val parity = metadata.getJSONObject("onnx_vs_torch_max_abs_error")
        for (head in parity.keys()) {
            assertTrue("$head: ${parity.getDouble(head)}", parity.getDouble(head) < 1e-4)
        }

        val cqt = metadata.getJSONObject("cqt")
        assertEquals(ChordFormerContract.SAMPLE_RATE, cqt.getInt("sample_rate"))
        assertEquals(ChordFormerContract.HOP_LENGTH, cqt.getInt("hop_length"))
        assertEquals(ChordFormerContract.BINS_PER_OCTAVE, cqt.getInt("bins_per_octave"))
        assertEquals(ChordFormerContract.ORIGINAL_HYBRID_BINS, cqt.getInt("original_bins"))
        assertEquals(ChordFormerContract.INPUT_BINS, cqt.getInt("input_bins"))
        assertEquals(
            ChordFormerContract.DROP_LOW_BINS,
            cqt.getJSONArray("model_crop").getInt(0),
        )
        assertTrue(
            cqt.getString("log_scaling"),
            cqt.getString("log_scaling").contains("amplitude_to_db"),
        )

        val decoder = metadata.getJSONObject("decoder")
        assertEquals(ChordFormerContract.VOCABULARY, decoder.getString("vocabulary"))
        assertEquals(
            ChordFormerContract.TRANSITION_PENALTY,
            decoder.getDouble("transition_penalty"),
            0.0,
        )
        assertEquals(301, decoder.getInt("candidate_count"))
        assertEquals(
            ChordFormerContract.DICTIONARY_SHA256,
            decoder.getString("dictionary_sha256"),
        )
    }

    @Test
    fun everyShippedAssetMatchesItsDeclaredDigest() {
        val metadata = JSONObject(
            ChordFormerTestAssets.locate(ChordFormerContract.METADATA_FILE)
                .readBytes()
                .toString(Charsets.UTF_8)
        )
        val declared = buildMap {
            put(metadata.getString("model_file"), metadata.getString("model_sha256"))
            val decoder = metadata.getJSONObject("decoder")
            put(decoder.getString("dictionary_file"), decoder.getString("dictionary_sha256"))
            val plans = metadata.getJSONObject("cqt").getJSONArray("plans")
            for (index in 0 until plans.length()) {
                val plan = plans.getJSONObject(index)
                put(plan.getString("file"), plan.getString("sha256"))
            }
        }
        assertEquals(4, declared.size)
        assertTrue(ChordFormerContract.LOW_CQT_PLAN_FILE in declared)
        assertTrue(ChordFormerContract.HIGH_CQT_PLAN_FILE in declared)

        for ((file, expected) in declared) {
            val bytes = ChordFormerTestAssets.locate(file).readBytes()
            assertEquals(file, expected, sha256(bytes))
        }
    }

    @Test
    fun theHeadLayoutIsTheOneTheDictionaryDecoderExpects() {
        assertEquals(
            100,
            ChordFormerContract.TRIAD_COUNT + ChordFormerContract.BASS_COUNT +
                ChordFormerContract.SEVENTH_COUNT + ChordFormerContract.NINTH_COUNT +
                ChordFormerContract.ELEVENTH_COUNT + ChordFormerContract.THIRTEENTH_COUNT,
        )
        assertEquals(
            ChordFormerContract.ORIGINAL_HYBRID_BINS,
            ChordFormerContract.RECURSIVE_BINS + ChordFormerContract.PSEUDO_BINS,
        )
        assertTrue(
            ChordFormerContract.DROP_LOW_BINS + ChordFormerContract.INPUT_BINS <=
                ChordFormerContract.ORIGINAL_HYBRID_BINS,
        )
        assertEquals(
            ChordFormerContract.OVERLAP_FRAMES,
            ChordFormerContract.HALF_OVERLAP_FRAMES * 2,
        )
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
