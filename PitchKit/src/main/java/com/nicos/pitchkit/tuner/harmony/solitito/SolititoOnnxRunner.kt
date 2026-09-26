package com.nicos.pitchkit.tuner.harmony.solitito

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * ONNX Runtime wrapper around solitito-ai `best_model_v2_take6`.
 *
 * One input, `features[1, 48, 168]`, and three heads: a 13-way root (twelve pitch classes
 * plus `Noise`), an 11-way quality, and twelve independent pitch logits. `brain.rs`
 * argmaxes and softmaxes the two categorical heads and multiplies their probabilities for
 * the confidence the vote weights by; the pitch head is sigmoided and, in the three-head
 * checkpoint the app ships, is carried for diagnostics only.
 *
 * The session is built exactly as the other three neural lanes build theirs, on ORT's own
 * default thread policy. Upstream pins `intra_op_num_threads = 1` (`brain.rs:86`) and that
 * is the wrong choice: the screening measured a 47.4 ms median forward pass at that
 * setting against 15.7 ms on four threads, which is a 1.18 duty cycle at the app's own
 * 40 ms cadence and cannot hold it. Nothing is pinned here, so solitito gets whatever the
 * device's ORT chooses, like every other lane; the documented lever if the duty cycle ever
 * bites is the cadence, not the pool — see [SolititoStreamingRecognizer].
 */
internal class SolititoOnnxRunner(
    modelBytes: ByteArray,
    expectedSha256: String = SolititoContract.MODEL_SHA256,
) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val hash = SolititoContract.sha256(modelBytes)
        require(hash == expectedSha256) { "Unexpected solitito model SHA-256: $hash" }
        // On a phone the default session spins one thread per core and still misses the
        // 40 ms cadence (18 % of capture buffers were dropped on a Pixel-class device).
        // A small fixed intra-op pool with sequential execution is the configuration the
        // other live lanes use and the one that keeps the cadence.
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(
            Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
        )
        options.setInterOpNumThreads(1)
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = environment.createSession(modelBytes, options)
        require(SolititoContract.INPUT_NAME in session.inputNames) {
            "solitito input '${SolititoContract.INPUT_NAME}' not found: ${session.inputNames}"
        }
        for (name in OUTPUTS) {
            require(name in session.outputNames) {
                "solitito output '$name' not found: ${session.outputNames}"
            }
        }
    }

    /** [window] is 48 x 168 features, oldest frame first. */
    @Synchronized
    fun predict(window: FloatArray): SolititoDecisionLayer.Reading {
        val expected = SolititoContract.CONTEXT_FRAMES * SolititoContract.FEATURE_COUNT
        require(window.size == expected) {
            "Expected $expected feature values, got ${window.size}"
        }
        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(window),
            longArrayOf(
                1L,
                SolititoContract.CONTEXT_FRAMES.toLong(),
                SolititoContract.FEATURE_COUNT.toLong(),
            ),
        ).use { input ->
            session.run(mapOf(SolititoContract.INPUT_NAME to input)).use { result ->
                val rootLogits = floats(
                    result, SolititoContract.ROOT_OUTPUT, SolititoContract.ROOT_CLASSES,
                )
                val qualityLogits = floats(
                    result, SolititoContract.QUALITY_OUTPUT, SolititoContract.QUALITY_CLASSES,
                )
                val pitchLogits = floats(
                    result, SolititoContract.PITCH_OUTPUT, SolititoContract.PITCH_CLASSES,
                )

                val (rootIndex, rootConfidence) = SolititoContract.argmaxSoftmax(rootLogits)
                val (qualityIndex, qualityConfidence) =
                    SolititoContract.argmaxSoftmax(qualityLogits)
                val quality = SolititoVocabulary.QUALITIES[
                    qualityIndex.coerceAtMost(SolititoVocabulary.QUALITIES.size - 1),
                ]
                val noise = rootIndex >= 12 || quality == "N"
                return SolititoDecisionLayer.Reading(
                    rootIndex = rootIndex,
                    quality = quality,
                    // `Noise` carries no confidence upstream, which is load-bearing: a
                    // vote history of nothing but noise never resolves to a label.
                    confidence = if (noise) 0.0 else rootConfidence * qualityConfidence,
                    pitches = FloatArray(pitchLogits.size) { index ->
                        (1.0 / (1.0 + exp(-pitchLogits[index].toDouble()))).toFloat()
                    },
                )
            }
        }
    }

    @Synchronized
    override fun close() {
        session.close()
    }

    private fun floats(result: OrtSession.Result, name: String, expected: Int): FloatArray {
        val value = result.get(name).orElseThrow {
            IllegalStateException("solitito output '$name' missing")
        }
        val tensor = value as? OnnxTensor ?: error("solitito output '$name' is not a tensor")
        val buffer = tensor.floatBuffer ?: error("solitito output '$name' is not float32")
        val values = FloatArray(buffer.remaining())
        buffer.get(values)
        require(values.size == expected) {
            "solitito output '$name' has ${values.size} values, expected $expected"
        }
        return values
    }

    private companion object {
        val OUTPUTS = listOf(
            SolititoContract.ROOT_OUTPUT,
            SolititoContract.QUALITY_OUTPUT,
            SolititoContract.PITCH_OUTPUT,
        )
    }
}
