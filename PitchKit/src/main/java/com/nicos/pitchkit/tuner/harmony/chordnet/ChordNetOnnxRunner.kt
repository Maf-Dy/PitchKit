package com.nicos.pitchkit.tuner.harmony.chordnet

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.security.MessageDigest

/** Thin ONNX Runtime wrapper around ChordMini ChordNet 2E1D. */
class ChordNetOnnxRunner(
    modelBytes: ByteArray,
) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val hash = sha256(modelBytes)
        require(hash == ChordNetContract.MODEL_SHA256) {
            "Unexpected ChordNet model SHA-256: $hash"
        }
        session = environment.createSession(modelBytes)
        require(ChordNetContract.INPUT_NAME in session.inputNames) {
            "ChordNet input '${ChordNetContract.INPUT_NAME}' not found: ${session.inputNames}"
        }
        require(ChordNetContract.OUTPUT_NAME in session.outputNames) {
            "ChordNet output '${ChordNetContract.OUTPUT_NAME}' not found: ${session.outputNames}"
        }
    }

    @Synchronized
    fun infer(features: FloatArray, windowCount: Int): FloatArray {
        require(windowCount > 0)
        val expected = windowCount * ChordNetContract.SEQUENCE_LENGTH * ChordNetContract.INPUT_BINS
        require(features.size == expected) {
            "Expected $expected feature values for $windowCount windows, got ${features.size}"
        }

        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(features),
            longArrayOf(
                windowCount.toLong(),
                ChordNetContract.SEQUENCE_LENGTH.toLong(),
                ChordNetContract.INPUT_BINS.toLong(),
            ),
        ).use { input ->
            session.run(mapOf(ChordNetContract.INPUT_NAME to input)).use { result ->
                val outputValue = result.get(ChordNetContract.OUTPUT_NAME)
                    .orElseThrow { IllegalStateException("ChordNet logits output missing") }
                val output = outputValue as? OnnxTensor
                    ?: error("ChordNet logits output is not a tensor")
                val buffer = output.floatBuffer
                    ?: error("ChordNet logits are not float32-compatible")
                val values = FloatArray(buffer.remaining())
                buffer.get(values)

                val expectedOutput = windowCount *
                    ChordNetContract.SEQUENCE_LENGTH *
                    ChordNetContract.CHORD_COUNT
                require(values.size == expectedOutput) {
                    "Expected $expectedOutput logits, got ${values.size}"
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
