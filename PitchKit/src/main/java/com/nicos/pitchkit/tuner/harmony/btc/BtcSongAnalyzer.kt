package com.nicos.pitchkit.tuner.harmony.btc

import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetVocabulary
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtCpuFrontend
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtPlanDecoder
import com.nicos.pitchkit.tuner.harmony.chordnet.StreamingPcmResampler
import com.nicos.pitchkit.tuner.harmony.song.SongChordSegment
import com.nicos.pitchkit.tuner.harmony.song.SongHarmonyAnalysis
import com.nicos.pitchkit.tuner.harmony.song.SongSectionDetector
import java.security.MessageDigest
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max

/**
 * Full-song BTC analyzer matching ChordMini's recommended offline evaluation path:
 * 50% overlap, Gaussian logit smoothing, logit aggregation, categorical smoothing,
 * and 0.5 s minimum segment duration.
 */
class BtcSongAnalyzer internal constructor(
    modelBytes: ByteArray,
    metadata: BtcMetadata,
    cqtPlanBytes: ByteArray,
    referenceA4Hz: Double = 440.0,
    private val preferFlats: Boolean = false,
) : AutoCloseable {
    private companion object {
        const val INFERENCE_BATCH_WINDOWS = 8
    }

    private val runner = BtcOnnxRunner(modelBytes, metadata.modelSha256)
    private val metadata = metadata
    private val frontend: CqtCpuFrontend
    private val resampler = StreamingPcmResampler(
        targetRate = BtcContract.SAMPLE_RATE,
        pitchScale = 440.0 / referenceA4Hz,
    )
    private val audio = FloatAccumulator(BtcContract.SAMPLE_RATE * 30)

    private var totalTargetSamples = 0L
    private var closed = false
    private var finished = false
    private var cachedResult: SongHarmonyAnalysis? = null

    init {
        require(referenceA4Hz in 300.0..600.0) {
            "referenceA4Hz must be between 300 and 600 Hz"
        }
        val planSha = sha256(cqtPlanBytes)
        require(planSha == BtcContract.PLAN_SHA256) {
            "Unexpected BTC CQT plan SHA-256: $planSha"
        }
        val plan = CqtPlanDecoder.decodeAndVerify(cqtPlanBytes)
        require(plan.config.sampleRate.toInt() == BtcContract.SAMPLE_RATE)
        require(plan.config.hopLength == BtcContract.HOP_LENGTH)
        require(plan.config.nBins == BtcContract.INPUT_BINS)
        require(plan.config.binsPerOctave == 24)
        require(plan.config.logMagnitude) { "BTC requires log-magnitude CQT features" }
        frontend = CqtCpuFrontend(plan)
    }

    /** Feed decoded mono PCM values in the -1..1 range. */
    @Synchronized
    fun accept(samples: FloatArray, sampleRate: Int) {
        check(!closed) { "BtcSongAnalyzer is closed" }
        check(!finished) { "BtcSongAnalyzer has already finished" }
        if (samples.isEmpty()) return
        val target = resampler.process(samples, sampleRate)
        if (target.isEmpty()) return
        totalTargetSamples += target.size
        audio.append(target)
    }

    @Synchronized
    fun finish(durationMs: Long = 0L): SongHarmonyAnalysis {
        check(!closed) { "BtcSongAnalyzer is closed" }
        cachedResult?.let { return it }
        finished = true

        val inferredDuration = totalTargetSamples * 1000L / BtcContract.SAMPLE_RATE
        val finalDuration = max(durationMs, inferredDuration).coerceAtLeast(0L)
        if (audio.size < BtcContract.HOP_LENGTH) {
            return SongHarmonyAnalysis(finalDuration, emptyList(), emptyList())
                .also { cachedResult = it }
        }

        val cqt = frontend.transform(audio.toArray())
        if (cqt.frameCount <= 0) {
            return SongHarmonyAnalysis(finalDuration, emptyList(), emptyList())
                .also { cachedResult = it }
        }

        val normalized = cqt.values.copyOf()
        val denominator = metadata.std.coerceAtLeast(1e-8f)
        for (index in normalized.indices) {
            normalized[index] = (normalized[index] - metadata.mean) / denominator
        }

        val predictions = inferSong(normalized, cqt.frameCount)
        val maxFramesFromDuration = if (finalDuration > 0L) {
            ((finalDuration / 1000.0) * BtcContract.SAMPLE_RATE / BtcContract.HOP_LENGTH)
                .toInt()
                .coerceAtLeast(1)
        } else {
            predictions.size
        }
        val validPredictions = predictions.take(minOf(predictions.size, maxFramesFromDuration))
        val chords = buildSegments(validPredictions, finalDuration)
        return SongHarmonyAnalysis(
            durationMs = finalDuration,
            chords = chords,
            sections = SongSectionDetector.detect(chords, finalDuration),
        ).also { cachedResult = it }
    }

    private data class FrameResult(
        val labelIndex: Int,
        val confidence: Double,
    )

    private fun inferSong(
        normalizedFeatures: FloatArray,
        frameCount: Int,
    ): List<FrameResult> {
        val seq = BtcContract.SEQUENCE_LENGTH
        val stride = BtcContract.WINDOW_STRIDE
        val paddedFrames = ceil(frameCount / seq.toDouble()).toInt().coerceAtLeast(1) * seq
        val starts = buildList {
            var start = 0
            while (start + seq <= paddedFrames) {
                add(start)
                start += stride
            }
        }

        val accumulator = FloatArray(frameCount * BtcContract.CHORD_COUNT)
        val counts = IntArray(frameCount)

        var batchStart = 0
        while (batchStart < starts.size) {
            val batchEnd = minOf(starts.size, batchStart + INFERENCE_BATCH_WINDOWS)
            val batchCount = batchEnd - batchStart
            val batch = FloatArray(batchCount * seq * BtcContract.INPUT_BINS)

            for (localWindow in 0 until batchCount) {
                val sourceStartFrame = starts[batchStart + localWindow]
                for (frame in 0 until seq) {
                    val sourceFrame = sourceStartFrame + frame
                    if (sourceFrame >= frameCount) break
                    val sourceOffset = sourceFrame * BtcContract.INPUT_BINS
                    val destinationOffset = (
                        localWindow * seq + frame
                    ) * BtcContract.INPUT_BINS
                    normalizedFeatures.copyInto(
                        destination = batch,
                        destinationOffset = destinationOffset,
                        startIndex = sourceOffset,
                        endIndex = sourceOffset + BtcContract.INPUT_BINS,
                    )
                }
            }

            val logits = runner.infer(batch, batchCount)
            val smoothed = gaussianSmoothLogits(logits, batchCount)

            for (localWindow in 0 until batchCount) {
                val globalStart = starts[batchStart + localWindow]
                for (frame in 0 until seq) {
                    val globalFrame = globalStart + frame
                    if (globalFrame >= frameCount) break
                    val sourceOffset = (
                        localWindow * seq + frame
                    ) * BtcContract.CHORD_COUNT
                    val destinationOffset = globalFrame * BtcContract.CHORD_COUNT
                    for (chord in 0 until BtcContract.CHORD_COUNT) {
                        accumulator[destinationOffset + chord] += smoothed[sourceOffset + chord]
                    }
                    counts[globalFrame]++
                }
            }
            batchStart = batchEnd
        }

        val rawIndices = IntArray(frameCount)
        val rawConfidence = DoubleArray(frameCount)
        for (frame in 0 until frameCount) {
            val count = counts[frame].coerceAtLeast(1)
            val offset = frame * BtcContract.CHORD_COUNT
            var best = 0
            var bestValue = accumulator[offset] / count
            for (chord in 1 until BtcContract.CHORD_COUNT) {
                val value = accumulator[offset + chord] / count
                if (value > bestValue) {
                    best = chord
                    bestValue = value
                }
            }
            rawIndices[frame] = best

            var denominator = 0.0
            for (chord in 0 until BtcContract.CHORD_COUNT) {
                denominator += exp((accumulator[offset + chord] / count - bestValue).toDouble())
            }
            rawConfidence[frame] = if (denominator > 0.0) 1.0 / denominator else 0.0
        }

        val filteredIndices = majorityFilter(rawIndices, BtcContract.SMOOTHING_KERNEL)
        return List(frameCount) { frame ->
            FrameResult(filteredIndices[frame], rawConfidence[frame])
        }
    }

    private fun gaussianSmoothLogits(
        logits: FloatArray,
        windowCount: Int,
    ): FloatArray {
        val kernelSize = BtcContract.SMOOTHING_KERNEL
        val radius = kernelSize / 2
        val sigma = kernelSize / 6.0
        val kernel = DoubleArray(kernelSize) { index ->
            val x = index - radius
            exp(-0.5 * (x / sigma) * (x / sigma))
        }
        val kernelSum = kernel.sum()
        for (index in kernel.indices) kernel[index] /= kernelSum

        val output = FloatArray(logits.size)
        val seq = BtcContract.SEQUENCE_LENGTH
        val classes = BtcContract.CHORD_COUNT
        for (window in 0 until windowCount) {
            for (frame in 0 until seq) {
                val destinationOffset = (window * seq + frame) * classes
                for (k in kernel.indices) {
                    val sourceFrame = (frame + k - radius).coerceIn(0, seq - 1)
                    val sourceOffset = (window * seq + sourceFrame) * classes
                    val weight = kernel[k]
                    for (chord in 0 until classes) {
                        output[destinationOffset + chord] += (logits[sourceOffset + chord] * weight).toFloat()
                    }
                }
            }
        }
        return output
    }

    private fun majorityFilter(values: IntArray, kernelSize: Int): IntArray {
        if (values.isEmpty() || kernelSize <= 1 || values.size < kernelSize) return values.copyOf()
        val radius = kernelSize / 2
        val output = IntArray(values.size)
        val counts = IntArray(BtcContract.CHORD_COUNT)
        for (frame in values.indices) {
            counts.fill(0)
            for (offset in -radius..radius) {
                val source = (frame + offset).coerceIn(0, values.lastIndex)
                counts[values[source]]++
            }
            var best = values[frame]
            for (label in counts.indices) {
                if (counts[label] > counts[best]) best = label
            }
            output[frame] = best
        }
        return output
    }

    private fun buildSegments(
        predictions: List<FrameResult>,
        durationMs: Long,
    ): List<SongChordSegment> {
        if (predictions.isEmpty()) return emptyList()
        val minimumFrames = ceil(
            BtcContract.MIN_SEGMENT_SECONDS * BtcContract.SAMPLE_RATE /
                BtcContract.HOP_LENGTH
        ).toInt().coerceAtLeast(1)

        val result = mutableListOf<SongChordSegment>()
        var start = 0
        while (start < predictions.size) {
            val labelIndex = predictions[start].labelIndex
            var end = start + 1
            while (end < predictions.size && predictions[end].labelIndex == labelIndex) end++

            val frameLength = end - start
            val rawLabel = ChordNetVocabulary.labelAt(labelIndex)
            if (frameLength >= minimumFrames && rawLabel != "N" && rawLabel != "X") {
                val startMs = frameToMs(start)
                val endMs = minOf(durationMs, frameToMs(end)).coerceAtLeast(startMs)
                if (endMs > startMs) {
                    result += SongChordSegment(
                        label = displayLabel(rawLabel),
                        startMs = startMs,
                        endMs = endMs,
                        confidence = predictions.subList(start, end)
                            .map { it.confidence }
                            .average(),
                    )
                }
            }
            start = end
        }
        return result
    }

    private fun displayLabel(raw: String): String {
        var display = ChordNetVocabulary.toDisplay(raw)
        if (!preferFlats) return display
        val flats = mapOf(
            "C#" to "Db",
            "D#" to "Eb",
            "F#" to "Gb",
            "G#" to "Ab",
            "A#" to "Bb",
        )
        for ((sharp, flat) in flats) {
            if (display.startsWith(sharp)) {
                display = flat + display.removePrefix(sharp)
                break
            }
        }
        return display
    }

    private fun frameToMs(frame: Int): Long =
        frame.toLong() * BtcContract.HOP_LENGTH * 1000L / BtcContract.SAMPLE_RATE

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        runner.close()
    }

    private class FloatAccumulator(initialCapacity: Int) {
        private var values = FloatArray(initialCapacity.coerceAtLeast(1))
        var size: Int = 0
            private set

        fun append(input: FloatArray) {
            ensureCapacity(size + input.size)
            input.copyInto(values, destinationOffset = size)
            size += input.size
        }

        fun toArray(): FloatArray = values.copyOf(size)

        private fun ensureCapacity(required: Int) {
            if (required <= values.size) return
            var capacity = values.size
            while (capacity < required) capacity *= 2
            values = values.copyOf(capacity)
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
