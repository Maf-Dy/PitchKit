package com.nicos.pitchkit.tuner.harmony.lvchordia

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp

internal data class LvChordiaHeads(
    val frames: Int,
    val triad: FloatArray,
    val bass: FloatArray,
    val seventh: FloatArray,
    val ninth: FloatArray,
    val eleventh: FloatArray,
    val thirteenth: FloatArray,
) {
    companion object {
        fun average(values: List<LvChordiaHeads>): LvChordiaHeads {
            require(values.isNotEmpty())
            val frames = values.first().frames
            require(values.all { it.frames == frames })
            return LvChordiaHeads(
                frames = frames,
                triad = averageArrays(values.map { it.triad }),
                bass = averageArrays(values.map { it.bass }),
                seventh = averageArrays(values.map { it.seventh }),
                ninth = averageArrays(values.map { it.ninth }),
                eleventh = averageArrays(values.map { it.eleventh }),
                thirteenth = averageArrays(values.map { it.thirteenth }),
            )
        }

        private fun averageArrays(inputs: List<FloatArray>): FloatArray {
            val size = inputs.first().size
            require(inputs.all { it.size == size })
            val result = FloatArray(size)
            for (input in inputs) {
                for (index in 0 until size) result[index] += input[index]
            }
            val divisor = inputs.size.toFloat()
            for (index in result.indices) result[index] /= divisor
            return result
        }
    }
}

/** Thin ONNX Runtime wrapper for one exported LV-Chordia ensemble member. */
internal class LvChordiaOnnxRunner(
    modelBytes: ByteArray,
) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = environment.createSession(modelBytes)
    private var closed = false

    init {
        require(LvChordiaContract.INPUT_NAME in session.inputNames) {
            "LV-Chordia input '${LvChordiaContract.INPUT_NAME}' missing: ${session.inputNames}"
        }
        val expected = setOf(
            LvChordiaContract.TRIAD_OUTPUT,
            LvChordiaContract.BASS_OUTPUT,
            LvChordiaContract.SEVENTH_OUTPUT,
            LvChordiaContract.NINTH_OUTPUT,
            LvChordiaContract.ELEVENTH_OUTPUT,
            LvChordiaContract.THIRTEENTH_OUTPUT,
        )
        require(session.outputNames.containsAll(expected)) {
            "Unexpected LV-Chordia outputs: ${session.outputNames}"
        }
    }

    @Synchronized
    fun infer(features: FloatArray, frameCount: Int): LvChordiaHeads {
        check(!closed) { "LV-Chordia OrtSession is closed" }
        require(frameCount > 0)
        require(features.size == frameCount * LvChordiaContract.INPUT_BINS) {
            "Expected ${frameCount * LvChordiaContract.INPUT_BINS} LV-Chordia values, got ${features.size}"
        }

        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(features),
            longArrayOf(1, frameCount.toLong(), LvChordiaContract.INPUT_BINS.toLong()),
        ).use { input ->
            session.run(mapOf(LvChordiaContract.INPUT_NAME to input)).use { result ->
                return LvChordiaHeads(
                    frames = frameCount,
                    triad = readProbabilities(result, LvChordiaContract.TRIAD_OUTPUT, frameCount, LvChordiaContract.TRIAD_COUNT),
                    bass = readProbabilities(result, LvChordiaContract.BASS_OUTPUT, frameCount, LvChordiaContract.BASS_COUNT),
                    seventh = readProbabilities(result, LvChordiaContract.SEVENTH_OUTPUT, frameCount, LvChordiaContract.SEVENTH_COUNT),
                    ninth = readProbabilities(result, LvChordiaContract.NINTH_OUTPUT, frameCount, LvChordiaContract.NINTH_COUNT),
                    eleventh = readProbabilities(result, LvChordiaContract.ELEVENTH_OUTPUT, frameCount, LvChordiaContract.ELEVENTH_COUNT),
                    thirteenth = readProbabilities(result, LvChordiaContract.THIRTEENTH_OUTPUT, frameCount, LvChordiaContract.THIRTEENTH_COUNT),
                )
            }
        }
    }

    private fun readProbabilities(
        result: OrtSession.Result,
        name: String,
        frames: Int,
        width: Int,
    ): FloatArray {
        val value = result.get(name)
            .orElseThrow { IllegalStateException("LV-Chordia output '$name' missing") }
        val tensor = value as? OnnxTensor
            ?: error("LV-Chordia output '$name' is not a tensor")
        val buffer = tensor.floatBuffer
            ?: error("LV-Chordia output '$name' is not float32")
        val logits = FloatArray(buffer.remaining())
        buffer.get(logits)
        require(logits.size == frames * width) {
            "LV-Chordia output '$name' expected ${frames * width} values, got ${logits.size}"
        }
        softmaxRows(logits, frames, width)
        return logits
    }

    private fun softmaxRows(values: FloatArray, rows: Int, width: Int) {
        for (row in 0 until rows) {
            val offset = row * width
            var maximum = Float.NEGATIVE_INFINITY
            for (index in 0 until width) maximum = maxOf(maximum, values[offset + index])
            var sum = 0.0
            for (index in 0 until width) {
                val probability = exp((values[offset + index] - maximum).toDouble())
                values[offset + index] = probability.toFloat()
                sum += probability
            }
            val divisor = sum.coerceAtLeast(1e-12).toFloat()
            for (index in 0 until width) values[offset + index] /= divisor
        }
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        session.close()
    }
}
