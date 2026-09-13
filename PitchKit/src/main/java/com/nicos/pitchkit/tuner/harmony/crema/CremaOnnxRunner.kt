package com.nicos.pitchkit.tuner.harmony.crema

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.security.MessageDigest

internal data class CremaHeads(
    val frames: Int,
    val tag: FloatArray,
    val pitch: FloatArray,
    val root: FloatArray,
    val bass: FloatArray,
)

/** Thin ONNX Runtime wrapper around the pinned Crema 0.2.0 conversion. */
internal class CremaOnnxRunner(
    modelBytes: ByteArray,
) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val hash = sha256(modelBytes)
        require(hash == CremaContract.MODEL_SHA256) {
            "Unexpected Crema model SHA-256: $hash"
        }
        session = environment.createSession(modelBytes)
        require(CremaContract.INPUT_NAME in session.inputNames) {
            "Crema input '${CremaContract.INPUT_NAME}' not found: ${session.inputNames}"
        }
        val requiredOutputs = setOf(
            CremaContract.TAG_OUTPUT,
            CremaContract.PITCH_OUTPUT,
            CremaContract.ROOT_OUTPUT,
            CremaContract.BASS_OUTPUT,
        )
        require(session.outputNames.containsAll(requiredOutputs)) {
            "Unexpected Crema outputs: ${session.outputNames}"
        }
    }

    @Synchronized
    fun infer(features: FloatArray, frameCount: Int): CremaHeads {
        require(frameCount > 0)
        val expected = frameCount * CremaContract.INPUT_BINS * CremaContract.HARMONIC_CHANNELS
        require(features.size == expected) {
            "Expected $expected Crema feature values, got ${features.size}"
        }

        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(features),
            longArrayOf(
                1,
                frameCount.toLong(),
                CremaContract.INPUT_BINS.toLong(),
                CremaContract.HARMONIC_CHANNELS.toLong(),
            ),
        ).use { input ->
            session.run(mapOf(CremaContract.INPUT_NAME to input)).use { result ->
                return CremaHeads(
                    frames = frameCount,
                    tag = readOutput(result, CremaContract.TAG_OUTPUT, frameCount, CremaContract.CHORD_COUNT),
                    pitch = readOutput(result, CremaContract.PITCH_OUTPUT, frameCount, CremaContract.PITCH_COUNT),
                    root = readOutput(result, CremaContract.ROOT_OUTPUT, frameCount, CremaContract.ROOT_COUNT),
                    bass = readOutput(result, CremaContract.BASS_OUTPUT, frameCount, CremaContract.BASS_COUNT),
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
        val outputValue = result.get(name)
            .orElseThrow { IllegalStateException("Crema output '$name' missing") }
        val tensor = outputValue as? OnnxTensor
            ?: error("Crema output '$name' is not a tensor")
        val buffer = tensor.floatBuffer
            ?: error("Crema output '$name' is not float32-compatible")
        val values = FloatArray(buffer.remaining())
        buffer.get(values)
        require(values.size == frames * width) {
            "Crema output '$name' expected ${frames * width} values, got ${values.size}"
        }
        return values
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
