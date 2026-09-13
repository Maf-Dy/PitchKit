package com.nicos.pitchkit.tuner.harmony.btc

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.security.MessageDigest

/** Thin ONNX Runtime wrapper around the ChordMini BTC continual-learning checkpoint. */
class BtcOnnxRunner(
    modelBytes: ByteArray,
    expectedSha256: String,
) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val hash = sha256(modelBytes)
        require(hash == expectedSha256.lowercase()) {
            "Unexpected BTC model SHA-256: $hash"
        }
        session = environment.createSession(modelBytes)
        require(BtcContract.INPUT_NAME in session.inputNames) {
            "BTC input '${BtcContract.INPUT_NAME}' not found: ${session.inputNames}"
        }
        require(BtcContract.OUTPUT_NAME in session.outputNames) {
            "BTC output '${BtcContract.OUTPUT_NAME}' not found: ${session.outputNames}"
        }
    }

    @Synchronized
    fun infer(features: FloatArray, windowCount: Int): FloatArray {
        require(windowCount > 0)
        val expected = windowCount * BtcContract.SEQUENCE_LENGTH * BtcContract.INPUT_BINS
        require(features.size == expected) {
            "Expected $expected BTC feature values for $windowCount windows, got ${features.size}"
        }

        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(features),
            longArrayOf(
                windowCount.toLong(),
                BtcContract.SEQUENCE_LENGTH.toLong(),
                BtcContract.INPUT_BINS.toLong(),
            ),
        ).use { input ->
            session.run(mapOf(BtcContract.INPUT_NAME to input)).use { result ->
                val outputValue = result.get(BtcContract.OUTPUT_NAME)
                    .orElseThrow { IllegalStateException("BTC logits output missing") }
                val output = outputValue as? OnnxTensor
                    ?: error("BTC logits output is not a tensor")
                val buffer = output.floatBuffer
                    ?: error("BTC logits are not float32-compatible")
                val values = FloatArray(buffer.remaining())
                buffer.get(values)

                val expectedOutput = windowCount *
                    BtcContract.SEQUENCE_LENGTH *
                    BtcContract.CHORD_COUNT
                require(values.size == expectedOutput) {
                    "Expected $expectedOutput BTC logits, got ${values.size}"
                }
                return values
            }
        }
    }

    @Synchronized
    override fun close() {
        session.close()
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
