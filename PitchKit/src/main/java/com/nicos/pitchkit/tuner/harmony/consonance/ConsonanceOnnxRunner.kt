package com.nicos.pitchkit.tuner.harmony.consonance

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.security.MessageDigest

internal data class ConsonanceHeads(
    val frames: Int,
    val root: FloatArray,
    val bass: FloatArray,
    val pitch: FloatArray,
)

internal class ConsonanceOnnxRunner(
    modelBytes: ByteArray,
    expectedSha256: String,
) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val hash = sha256(modelBytes)
        require(hash == expectedSha256.lowercase()) {
            "Unexpected Consonance model SHA-256: $hash"
        }
        session = environment.createSession(modelBytes)
        require(ConsonanceContract.INPUT_NAME in session.inputNames) {
            "Consonance input '${ConsonanceContract.INPUT_NAME}' not found: ${session.inputNames}"
        }
        val requiredOutputs = setOf(
            ConsonanceContract.ROOT_OUTPUT,
            ConsonanceContract.BASS_OUTPUT,
            ConsonanceContract.CHORD_OUTPUT,
        )
        require(session.outputNames.containsAll(requiredOutputs)) {
            "Unexpected Consonance outputs: ${session.outputNames}"
        }
    }

    /** Input is fixed feature-major [144,862], matching ONNX [1,1,144,862]. */
    @Synchronized
    fun infer(featureMajor: FloatArray): ConsonanceHeads {
        val frames = ConsonanceContract.SEQUENCE_FRAMES
        val expected = ConsonanceContract.INPUT_BINS * frames
        require(featureMajor.size == expected) {
            "Expected $expected Consonance values for fixed ${frames}-frame input, got ${featureMajor.size}"
        }

        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(featureMajor),
            longArrayOf(
                1,
                1,
                ConsonanceContract.INPUT_BINS.toLong(),
                frames.toLong(),
            ),
        ).use { input ->
            session.run(mapOf(ConsonanceContract.INPUT_NAME to input)).use { result ->
                return ConsonanceHeads(
                    frames = frames,
                    root = readOutput(
                        result,
                        ConsonanceContract.ROOT_OUTPUT,
                        frames,
                        ConsonanceContract.ROOT_COUNT,
                    ),
                    bass = readOutput(
                        result,
                        ConsonanceContract.BASS_OUTPUT,
                        frames,
                        ConsonanceContract.BASS_COUNT,
                    ),
                    pitch = readOutput(
                        result,
                        ConsonanceContract.CHORD_OUTPUT,
                        frames,
                        ConsonanceContract.PITCH_COUNT,
                    ),
                )
            }
        }
    }

    private fun readOutput(
        result: OrtSession.Result,
        name: String,
        frames: Int,
        width: Int,
    ): FloatArray {
        val value = result.get(name)
            .orElseThrow { IllegalStateException("Consonance output '$name' missing") }
        val tensor = value as? OnnxTensor
            ?: error("Consonance output '$name' is not a tensor")
        val buffer = tensor.floatBuffer
            ?: error("Consonance output '$name' is not float32-compatible")
        val output = FloatArray(buffer.remaining())
        buffer.get(output)
        require(output.size == frames * width) {
            "Consonance output '$name' expected ${frames * width} values, got ${output.size}"
        }
        return output
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
