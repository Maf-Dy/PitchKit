package com.nicos.pitchkit.tuner.harmony.chordformer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaHeads
import java.nio.FloatBuffer

/**
 * Thin ONNX Runtime wrapper for the exported ChordFormer fold-3 graph.
 *
 * The export wraps the six heads in a softmax, so the outputs are already per-frame
 * probabilities and are reused as LV Song's head bundle: the layout is identical
 * (73/13/4/4/3/3) and the dictionary XHMM decoder downstream is LV Song's.
 *
 * The graph is traced at a fixed 2048-frame length, so `infer` only accepts that length.
 */
internal class ChordFormerOnnxRunner(
    modelBytes: ByteArray,
) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = environment.createSession(modelBytes)
    private var closed = false

    init {
        require(ChordFormerContract.INPUT_NAME in session.inputNames) {
            "ChordFormer input '${ChordFormerContract.INPUT_NAME}' missing: ${session.inputNames}"
        }
        val expected = setOf(
            ChordFormerContract.TRIAD_OUTPUT,
            ChordFormerContract.BASS_OUTPUT,
            ChordFormerContract.SEVENTH_OUTPUT,
            ChordFormerContract.NINTH_OUTPUT,
            ChordFormerContract.ELEVENTH_OUTPUT,
            ChordFormerContract.THIRTEENTH_OUTPUT,
        )
        require(session.outputNames.containsAll(expected)) {
            "Unexpected ChordFormer outputs: ${session.outputNames}"
        }
    }

    @Synchronized
    fun infer(features: FloatArray): LvChordiaHeads {
        check(!closed) { "ChordFormer OrtSession is closed" }
        val frames = ChordFormerContract.WINDOW_FRAMES
        require(features.size == frames * ChordFormerContract.INPUT_BINS) {
            "ChordFormer expects ${frames * ChordFormerContract.INPUT_BINS} values, got ${features.size}"
        }

        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(features),
            longArrayOf(1, frames.toLong(), ChordFormerContract.INPUT_BINS.toLong()),
        ).use { input ->
            session.run(mapOf(ChordFormerContract.INPUT_NAME to input)).use { result ->
                return LvChordiaHeads(
                    frames = frames,
                    triad = read(result, ChordFormerContract.TRIAD_OUTPUT, frames, ChordFormerContract.TRIAD_COUNT),
                    bass = read(result, ChordFormerContract.BASS_OUTPUT, frames, ChordFormerContract.BASS_COUNT),
                    seventh = read(result, ChordFormerContract.SEVENTH_OUTPUT, frames, ChordFormerContract.SEVENTH_COUNT),
                    ninth = read(result, ChordFormerContract.NINTH_OUTPUT, frames, ChordFormerContract.NINTH_COUNT),
                    eleventh = read(result, ChordFormerContract.ELEVENTH_OUTPUT, frames, ChordFormerContract.ELEVENTH_COUNT),
                    thirteenth = read(result, ChordFormerContract.THIRTEENTH_OUTPUT, frames, ChordFormerContract.THIRTEENTH_COUNT),
                )
            }
        }
    }

    private fun read(
        result: OrtSession.Result,
        name: String,
        frames: Int,
        width: Int,
    ): FloatArray {
        val value = result.get(name)
            .orElseThrow { IllegalStateException("ChordFormer output '$name' missing") }
        val tensor = value as? OnnxTensor ?: error("ChordFormer output '$name' is not a tensor")
        val buffer = tensor.floatBuffer ?: error("ChordFormer output '$name' is not float32")
        val values = FloatArray(buffer.remaining())
        buffer.get(values)
        require(values.size == frames * width) {
            "ChordFormer output '$name' expected ${frames * width} values, got ${values.size}"
        }
        return values
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        session.close()
    }
}
