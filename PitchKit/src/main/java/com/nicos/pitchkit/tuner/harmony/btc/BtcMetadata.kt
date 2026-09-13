package com.nicos.pitchkit.tuner.harmony.btc

import java.io.ByteArrayInputStream
import java.util.Properties

data class BtcMetadata(
    val mean: Float,
    val std: Float,
    val modelSha256: String,
    val sourceCommit: String,
    val checkpointBlobSha: String,
) {
    companion object {
        fun parse(bytes: ByteArray): BtcMetadata {
            val properties = Properties().apply {
                ByteArrayInputStream(bytes).use { input -> load(input) }
            }
            fun required(name: String): String = properties.getProperty(name)
                ?.trim()
                ?.takeIf { it.isNotEmpty() }
                ?: error("BTC metadata property '$name' is missing")

            val mean = required("mean").toFloat()
            val std = required("std").toFloat()
            require(std > 0f) { "BTC normalization std must be positive" }

            val modelSha256 = required("model_sha256").lowercase()
            require(modelSha256.length == 64 && modelSha256.all { it in "0123456789abcdef" }) {
                "BTC model SHA-256 in metadata is invalid"
            }

            return BtcMetadata(
                mean = mean,
                std = std,
                modelSha256 = modelSha256,
                sourceCommit = required("source_commit"),
                checkpointBlobSha = required("checkpoint_blob_sha"),
            )
        }
    }
}
