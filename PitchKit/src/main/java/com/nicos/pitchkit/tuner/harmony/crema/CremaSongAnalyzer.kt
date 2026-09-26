package com.nicos.pitchkit.tuner.harmony.crema

import com.nicos.pitchkit.tuner.harmony.chordnet.StreamingPcmResampler
import com.nicos.pitchkit.tuner.harmony.song.SongChordSegment
import com.nicos.pitchkit.tuner.harmony.song.SongHarmonyAnalysis
import com.nicos.pitchkit.tuner.harmony.song.SongSectionDetector
import kotlin.math.max
import kotlin.math.min

/**
 * Whole-song Crema analyzer. PCM is accepted incrementally, the HCQT is built in
 * bounded windows as the audio streams in, and the accumulated feature frames are
 * then run through the model and decoded once at the end with Crema's transition
 * statistics instead of the live heuristic. See [CremaInferencePlanner] for why
 * inference is a single whole-song pass.
 */
class CremaSongAnalyzer internal constructor(
    modelBytes: ByteArray,
    runtimeStateJson: String,
    harmonic1PlanBytes: ByteArray,
    harmonic2PlanBytes: ByteArray,
    referenceA4Hz: Double = 440.0,
    preferFlats: Boolean = false,
) : AutoCloseable {
    private companion object {
        // HCQT windowing only: these bound how much PCM is transformed at once.
        // The model's context is chosen by CremaInferencePlanner, not by these.
        const val WINDOW_FRAMES = 128
        const val OVERLAP_FRAMES = 16
        const val HALF_OVERLAP_FRAMES = OVERLAP_FRAMES / 2
        const val STEP_FRAMES = WINDOW_FRAMES - OVERLAP_FRAMES
        const val MIN_FINAL_FRAMES = 8
        const val FEATURE_VALUES_PER_FRAME =
            CremaContract.INPUT_BINS * CremaContract.HARMONIC_CHANNELS
    }

    private val frontend = CremaHcqtFrontend(harmonic1PlanBytes, harmonic2PlanBytes)
    private val runner = CremaOnnxRunner(modelBytes)
    private val runtimeState = CremaRuntimeState.parse(runtimeStateJson)
    private val sequenceDecoder = CremaSongViterbiDecoder(
        state = runtimeState,
        preferFlats = preferFlats,
    )
    private val resampler = StreamingPcmResampler(
        targetRate = CremaContract.SAMPLE_RATE,
        pitchScale = 440.0 / referenceA4Hz,
    )
    private val buffer = FloatSlidingBuffer(WINDOW_FRAMES * CremaContract.HOP_LENGTH * 2)

    /** Committed HCQT frames for the whole song, fed to the model in [finish]. */
    private val features = FloatSlidingBuffer(WINDOW_FRAMES * FEATURE_VALUES_PER_FRAME)

    private var windowStartFrame = 0L
    private var totalTargetSamples = 0L
    private var closed = false
    private var finished = false
    private var cachedResult: SongHarmonyAnalysis? = null

    init {
        require(referenceA4Hz in 300.0..600.0) {
            "referenceA4Hz must be between 300 and 600 Hz"
        }
    }

    @Synchronized
    fun accept(samples: FloatArray, sampleRate: Int) {
        check(!closed) { "CremaSongAnalyzer is closed" }
        check(!finished) { "CremaSongAnalyzer has already finished" }
        if (samples.isEmpty()) return

        val target = resampler.process(samples, sampleRate)
        if (target.isEmpty()) return
        totalTargetSamples += target.size
        buffer.append(target)

        val windowSamples = WINDOW_FRAMES * CremaContract.HOP_LENGTH
        val stepSamples = STEP_FRAMES * CremaContract.HOP_LENGTH
        while (buffer.size >= windowSamples) {
            extractWindowFeatures(
                audio = buffer.copyFirst(windowSamples),
                finalWindow = false,
            )
            buffer.dropFirst(stepSamples)
            windowStartFrame += STEP_FRAMES
        }
    }

    @Synchronized
    fun finish(durationMs: Long = 0L): SongHarmonyAnalysis {
        check(!closed) { "CremaSongAnalyzer is closed" }
        cachedResult?.let { return it }
        if (!finished) {
            val remainingFrames = buffer.size / CremaContract.HOP_LENGTH
            if (remainingFrames >= MIN_FINAL_FRAMES) {
                extractWindowFeatures(
                    audio = buffer.copyFirst(remainingFrames * CremaContract.HOP_LENGTH),
                    finalWindow = true,
                )
            }
            runInference()
            finished = true
        }

        val inferredDuration = (
            totalTargetSamples * 1000L / CremaContract.SAMPLE_RATE
        ).coerceAtLeast(0L)
        val finalDuration = max(durationMs, inferredDuration)
        val chords = buildChordSegments(finalDuration)
        return SongHarmonyAnalysis(
            durationMs = finalDuration,
            chords = chords,
            sections = SongSectionDetector.detect(chords, finalDuration),
        ).also { cachedResult = it }
    }

    /**
     * Streams the HCQT: only [WINDOW_FRAMES] of PCM are transformed at a time and
     * the overlap edges are dropped, so the kept frames are contiguous. No model
     * inference happens here - see [runInference].
     */
    private fun extractWindowFeatures(audio: FloatArray, finalWindow: Boolean) {
        val window = frontend.transform(audio)
        if (window.frameCount <= 0) return

        val commitStart = if (windowStartFrame == 0L) 0 else HALF_OVERLAP_FRAMES
        val commitEnd = if (finalWindow) {
            window.frameCount
        } else {
            min(window.frameCount, WINDOW_FRAMES - HALF_OVERLAP_FRAMES)
        }
        if (commitStart >= commitEnd) return

        features.append(
            window.values,
            from = commitStart * FEATURE_VALUES_PER_FRAME,
            to = commitEnd * FEATURE_VALUES_PER_FRAME,
        )
    }

    private fun runInference() {
        val frameCount = features.size / FEATURE_VALUES_PER_FRAME
        if (frameCount <= 0) return

        val plan = CremaInferencePlanner.plan(frameCount)
        if (plan.size == 1 && frameCount > CremaInferencePlanner.FALLBACK_CHUNK_FRAMES) {
            val failed = try {
                inferChunks(plan)
                false
            } catch (_: OutOfMemoryError) {
                true
            } catch (_: RuntimeException) {
                true
            }
            // The single pass commits nothing until it returns, so the windowed
            // path can still run the whole song from the same features.
            if (failed) inferChunks(CremaInferencePlanner.chunked(frameCount))
        } else {
            inferChunks(plan)
        }
        features.clear()
    }

    private fun inferChunks(chunks: List<CremaInferenceChunk>) {
        for (chunk in chunks) {
            val values = features.copyRange(
                from = chunk.startFrame * FEATURE_VALUES_PER_FRAME,
                to = chunk.endFrame * FEATURE_VALUES_PER_FRAME,
            )
            val heads = runner.infer(values, chunk.frameCount)
            for (frame in chunk.commitStart until chunk.commitEnd) {
                sequenceDecoder.add(
                    heads = heads,
                    localFrame = frame - chunk.startFrame,
                    globalFrame = frame.toLong(),
                )
            }
        }
    }

    private fun buildChordSegments(durationMs: Long): List<SongChordSegment> {
        val ordered = sequenceDecoder.decode()
        if (ordered.isEmpty()) return emptyList()

        val raw = mutableListOf<SongChordSegment>()
        var segmentStart = 0
        while (segmentStart < ordered.size) {
            val label = ordered[segmentStart].label
            var segmentEnd = segmentStart + 1
            while (segmentEnd < ordered.size && ordered[segmentEnd].label == label) {
                segmentEnd++
            }

            if (label != null) {
                val startMs = frameToMs(ordered[segmentStart].frame)
                val nextFrame = ordered.getOrNull(segmentEnd)?.frame
                val endMs = if (nextFrame != null) frameToMs(nextFrame) else durationMs
                val confidence = ordered.subList(segmentStart, segmentEnd)
                    .map { it.confidence }
                    .averageOrZero()
                raw += SongChordSegment(
                    label = label,
                    startMs = startMs.coerceAtMost(durationMs),
                    endMs = max(startMs, endMs).coerceAtMost(durationMs),
                    confidence = confidence,
                )
            }
            segmentStart = segmentEnd
        }
        return mergeAdjacent(raw.filter { it.endMs > it.startMs })
    }

    private fun mergeAdjacent(input: List<SongChordSegment>): List<SongChordSegment> {
        if (input.isEmpty()) return emptyList()
        val result = mutableListOf<SongChordSegment>()
        for (segment in input) {
            val previous = result.lastOrNull()
            val gap = if (previous != null) segment.startMs - previous.endMs else Long.MAX_VALUE
            if (previous != null && previous.label == segment.label && gap <= 350L) {
                val previousDuration = max(1L, previous.endMs - previous.startMs)
                val segmentDuration = max(1L, segment.endMs - segment.startMs)
                val weightedConfidence = (
                    previous.confidence * previousDuration +
                        segment.confidence * segmentDuration
                ) / (previousDuration + segmentDuration).toDouble()
                result[result.lastIndex] = previous.copy(
                    endMs = segment.endMs,
                    confidence = weightedConfidence,
                )
            } else {
                result += segment
            }
        }
        return result
    }

    private fun frameToMs(frame: Long): Long =
        frame * CremaContract.HOP_LENGTH * 1000L / CremaContract.SAMPLE_RATE

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        frontend.close()
        runner.close()
    }

    private class FloatSlidingBuffer(initialCapacity: Int) {
        private var values = FloatArray(initialCapacity.coerceAtLeast(1))
        var size: Int = 0
            private set

        fun append(input: FloatArray) = append(input, 0, input.size)

        fun append(input: FloatArray, from: Int, to: Int) {
            require(from in 0..to && to <= input.size)
            val count = to - from
            if (count == 0) return
            ensureCapacity(size + count)
            input.copyInto(values, destinationOffset = size, startIndex = from, endIndex = to)
            size += count
        }

        fun copyFirst(count: Int): FloatArray {
            require(count in 0..size)
            return values.copyOfRange(0, count)
        }

        fun copyRange(from: Int, to: Int): FloatArray {
            require(from in 0..to && to <= size)
            return values.copyOfRange(from, to)
        }

        fun clear() {
            size = 0
            values = FloatArray(1)
        }

        fun dropFirst(count: Int) {
            val actual = count.coerceIn(0, size)
            if (actual == 0) return
            values.copyInto(
                destination = values,
                destinationOffset = 0,
                startIndex = actual,
                endIndex = size,
            )
            size -= actual
        }

        private fun ensureCapacity(required: Int) {
            if (required <= values.size) return
            var capacity = values.size
            while (capacity < required) capacity *= 2
            values = values.copyOf(capacity)
        }
    }
}

private fun List<Double>.averageOrZero(): Double = if (isEmpty()) 0.0 else average()
