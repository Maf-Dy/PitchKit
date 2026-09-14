package com.nicos.pitchkit.tuner.harmony.consonance

import org.json.JSONObject

data class ConsonanceMetadata(
    val modelSha256: String,
    val modelSize: Long,
    val sourceCommit: String,
) {
    companion object {
        fun parse(bytes: ByteArray): ConsonanceMetadata {
            val json = JSONObject(bytes.toString(Charsets.UTF_8))
            require(json.getInt("schema_version") == 1) { "Unsupported Consonance metadata schema" }
            val sourceCommit = json.getString("source_commit")
            require(sourceCommit == ConsonanceContract.SOURCE_COMMIT) {
                "Unexpected Consonance source commit: $sourceCommit"
            }
            require(json.getInt("sample_rate") == ConsonanceContract.SAMPLE_RATE)
            require(json.getInt("hop_length") == ConsonanceContract.HOP_LENGTH)
            require(json.getInt("chunk_seconds") == ConsonanceContract.CHUNK_SECONDS)
            require(json.getInt("sequence_frames") == ConsonanceContract.SEQUENCE_FRAMES)
            require(json.getInt("input_bins") == ConsonanceContract.INPUT_BINS)
            require(json.getInt("root_classes") == ConsonanceContract.ROOT_COUNT)
            require(json.getInt("bass_classes") == ConsonanceContract.BASS_COUNT)
            require(json.getInt("pitch_classes") == ConsonanceContract.PITCH_COUNT)

            val sha = json.getString("model_sha256").lowercase()
            require(sha.length == 64 && sha.all { it in "0123456789abcdef" }) {
                "Invalid Consonance model SHA-256"
            }
            return ConsonanceMetadata(
                modelSha256 = sha,
                modelSize = json.getLong("model_size"),
                sourceCommit = sourceCommit,
            )
        }
    }
}
