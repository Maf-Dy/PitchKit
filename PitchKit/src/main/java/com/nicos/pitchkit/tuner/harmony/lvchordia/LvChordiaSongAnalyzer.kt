package com.nicos.pitchkit.tuner.harmony.lvchordia

import com.nicos.pitchkit.tuner.harmony.chordnet.StreamingPcmResampler
import com.nicos.pitchkit.tuner.harmony.song.SongChordSegment
import com.nicos.pitchkit.tuner.harmony.song.SongHarmonyAnalysis
import com.nicos.pitchkit.tuner.harmony.song.SongSection
import com.nicos.pitchkit.tuner.harmony.song.SongSectionDetector
import kotlin.math.max
import kotlin.math.min

/**
 * Offline LV Song large-vocabulary analyzer.
 *
 * Upstream LV-Chordia runs a bidirectional LSTM over the song CQT before the
 * dictionary HMM. Running tiny independent windows is therefore not equivalent:
 * frames near each artificial window boundary lose most of their future/past
 * context and ambiguous chord changes can be reported late.
 *
 * Android cannot safely run an arbitrarily long full-song BiLSTM in one ONNX
 * call, so we use overlap-protected windows. Only the well-contextualized center
 * of each internal window is committed. Ensemble members are evaluated
 * sequentially to keep peak activation memory bounded.
 */
class LvChordiaSongAnalyzer internal constructor(
    modelBytes: List<ByteArray>,
    dictionaryJson: String,
    lowPlanBytes: ByteArray,
    highPlanBytes: ByteArray,
    referenceA4Hz: Double = 440.0,
    preferFlats: Boolean = false,
) : AutoCloseable {
    private companion object {
        // 1024 frames = ~23.8 s at 22.05 kHz / 512 hop.
        // 384-frame overlap leaves ~4.46 s of protected context on each side
        // of an internal commit boundary, versus ~0.74 s in the old port.
        const val MODEL_WINDOW_FRAMES = 1024
        const val MODEL_OVERLAP_FRAMES = 384
        const val MODEL_HALF_OVERLAP = MODEL_OVERLAP_FRAMES / 2
        const val MODEL_STEP_FRAMES = MODEL_WINDOW_FRAMES - MODEL_OVERLAP_FRAMES
    }

    private val frontend = LvChordiaHybridCqtFrontend(lowPlanBytes, highPlanBytes)
    private val runners = modelBytes.map(::LvChordiaOnnxRunner)
    private val dictionary = LvChordiaDictionaryParser.parse(dictionaryJson, preferFlats)
    private val sequenceDecoder = LvChordiaDictionaryDecoder(dictionary)
    private val resampler = StreamingPcmResampler(
        targetRate = LvChordiaContract.SAMPLE_RATE,
        pitchScale = 440.0 / referenceA4Hz,
    )
    private val pcm = FloatAccumulator(initialCapacity = LvChordiaContract.SAMPLE_RATE * 30)

    private var totalTargetSamples = 0L
    private var closed = false
    private var finished = false
    private var cachedResult: SongHarmonyAnalysis? = null

    init {
        require(modelBytes.size == LvChordiaContract.MODEL_FILES.size) {
            "Songs mode requires all ${LvChordiaContract.MODEL_FILES.size} LV Song ensemble members"
        }
        require(referenceA4Hz in 300.0..600.0)
        require(MODEL_OVERLAP_FRAMES < MODEL_WINDOW_FRAMES)
        require(MODEL_HALF_OVERLAP * 2 == MODEL_OVERLAP_FRAMES)
    }

    /** Feed decoded mono PCM values in the -1..1 range. */
    @Synchronized
    fun accept(samples: FloatArray, sampleRate: Int) {
        check(!closed) { "LvChordiaSongAnalyzer is closed" }
        check(!finished) { "LvChordiaSongAnalyzer has already finished" }
        if (samples.isEmpty()) return

        val target = resampler.process(samples, sampleRate)
        if (target.isEmpty()) return
        totalTargetSamples += target.size
        pcm.append(target)
    }

    @Synchronized
    fun finish(durationMs: Long = 0L): SongHarmonyAnalysis {
        check(!closed) { "LvChordiaSongAnalyzer is closed" }
        cachedResult?.let { return it }
        check(!finished) { "LvChordiaSongAnalyzer has already finished" }
        finished = true

        val inferredDuration = totalTargetSamples * 1000L / LvChordiaContract.SAMPLE_RATE
        val finalDuration = max(durationMs, inferredDuration)
        if (pcm.size < LvChordiaContract.HOP_LENGTH) {
            return emptyAnalysis(finalDuration).also { cachedResult = it }
        }

        val features = frontend.transform(pcm.toFloatArray())
        pcm.clear()
        if (features.frameCount <= 0) {
            return emptyAnalysis(finalDuration).also { cachedResult = it }
        }

        val committed = HeadsAccumulator()
        val frameNumbers = LongAccumulator(initialCapacity = features.frameCount)
        analyzeFeatureWindows(features, committed, frameNumbers)

        val heads = committed.build()
        if (heads.frames <= 0) {
            return emptyAnalysis(finalDuration).also { cachedResult = it }
        }
        val frames = frameNumbers.toLongArray()
        require(frames.size == heads.frames)

        // Keep the upstream authority chain intact:
        // CQT -> 5-model probability average -> full dictionary HMM -> segments.
        val decoded = sequenceDecoder.decode(heads)
        require(decoded.size == frames.size)

        val chords = buildChordSegments(decoded, frames, finalDuration)
        return SongHarmonyAnalysis(
            durationMs = finalDuration,
            chords = chords,
            sections = SongSectionDetector.detect(chords, finalDuration),
        ).also { cachedResult = it }
    }

    private fun analyzeFeatureWindows(
        features: LvChordiaFeatures,
        committed: HeadsAccumulator,
        frameNumbers: LongAccumulator,
    ) {
        var startFrame = 0
        while (startFrame < features.frameCount) {
            val endFrame = min(features.frameCount, startFrame + MODEL_WINDOW_FRAMES)
            val frameCount = endFrame - startFrame
            val windowValues = FloatArray(frameCount * LvChordiaContract.INPUT_BINS)
            features.values.copyInto(
                destination = windowValues,
                startIndex = startFrame * LvChordiaContract.INPUT_BINS,
                endIndex = endFrame * LvChordiaContract.INPUT_BINS,
            )

            // Do not run several large BiLSTM/CNN graphs concurrently. The
            // upstream ensemble average is mathematically identical whether
            // members are evaluated concurrently or sequentially, while the
            // latter has a much safer Android peak-memory profile.
            val average = LvChordiaHeads.average(
                runners.map { runner -> runner.infer(windowValues, frameCount) }
            )

            val isFirst = startFrame == 0
            val isLast = endFrame == features.frameCount
            val localCommitStart = if (isFirst) 0 else MODEL_HALF_OVERLAP
            val localCommitEnd = if (isLast) {
                frameCount
            } else {
                (frameCount - MODEL_HALF_OVERLAP).coerceAtLeast(localCommitStart)
            }

            if (localCommitStart < localCommitEnd) {
                committed.append(average, localCommitStart, localCommitEnd)
                for (local in localCommitStart until localCommitEnd) {
                    frameNumbers.add((startFrame + local).toLong())
                }
            }

            if (isLast) break
            startFrame += MODEL_STEP_FRAMES
        }
    }

    private fun emptyAnalysis(durationMs: Long): SongHarmonyAnalysis = SongHarmonyAnalysis(
        durationMs = durationMs,
        chords = emptyList(),
        sections = if (durationMs > 0L) listOf(SongSection("A", 0L, durationMs)) else emptyList(),
    )

    private fun buildChordSegments(
        decoded: List<LvChordiaDecodedFrame>,
        frames: LongArray,
        durationMs: Long,
    ): List<SongChordSegment> {
        val result = mutableListOf<SongChordSegment>()
        var start = 0
        while (start < decoded.size) {
            val label = decoded[start].label
            var end = start + 1
            while (end < decoded.size && decoded[end].label == label) end++

            if (label != null) {
                val startMs = frameToMs(frames[start]).coerceAtMost(durationMs)
                val endMs = if (end < frames.size) {
                    frameToMs(frames[end]).coerceAtMost(durationMs)
                } else {
                    durationMs
                }
                if (endMs > startMs) {
                    result += SongChordSegment(
                        label = label,
                        startMs = startMs,
                        endMs = endMs,
                        confidence = decoded.subList(start, end)
                            .map { it.confidence }
                            .averageOrZero(),
                    )
                }
            }
            start = end
        }
        return mergeAdjacent(result)
    }

    private fun mergeAdjacent(input: List<SongChordSegment>): List<SongChordSegment> {
        if (input.isEmpty()) return input
        val result = mutableListOf<SongChordSegment>()
        for (segment in input) {
            val previous = result.lastOrNull()
            val gap = if (previous == null) Long.MAX_VALUE else segment.startMs - previous.endMs
            if (previous != null && previous.label == segment.label && gap <= 150L) {
                val firstDuration = max(1L, previous.endMs - previous.startMs)
                val secondDuration = max(1L, segment.endMs - segment.startMs)
                val confidence = (
                    previous.confidence * firstDuration + segment.confidence * secondDuration
                ) / (firstDuration + secondDuration).toDouble()
                result[result.lastIndex] = previous.copy(
                    endMs = segment.endMs,
                    confidence = confidence,
                )
            } else {
                result += segment
            }
        }
        return result
    }

    private fun frameToMs(frame: Long): Long =
        frame * LvChordiaContract.HOP_LENGTH * 1000L / LvChordiaContract.SAMPLE_RATE

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        runners.forEach { runCatching { it.close() } }
        pcm.clear()
    }

    private class HeadsAccumulator {
        private val triad = FloatAccumulator()
        private val bass = FloatAccumulator()
        private val seventh = FloatAccumulator()
        private val ninth = FloatAccumulator()
        private val eleventh = FloatAccumulator()
        private val thirteenth = FloatAccumulator()
        private var frames = 0

        fun append(heads: LvChordiaHeads, firstFrame: Int, endFrame: Int) {
            if (firstFrame >= endFrame) return
            triad.appendRows(heads.triad, LvChordiaContract.TRIAD_COUNT, firstFrame, endFrame)
            bass.appendRows(heads.bass, LvChordiaContract.BASS_COUNT, firstFrame, endFrame)
            seventh.appendRows(heads.seventh, LvChordiaContract.SEVENTH_COUNT, firstFrame, endFrame)
            ninth.appendRows(heads.ninth, LvChordiaContract.NINTH_COUNT, firstFrame, endFrame)
            eleventh.appendRows(heads.eleventh, LvChordiaContract.ELEVENTH_COUNT, firstFrame, endFrame)
            thirteenth.appendRows(heads.thirteenth, LvChordiaContract.THIRTEENTH_COUNT, firstFrame, endFrame)
            frames += endFrame - firstFrame
        }

        fun build(): LvChordiaHeads = LvChordiaHeads(
            frames = frames,
            triad = triad.toFloatArray(),
            bass = bass.toFloatArray(),
            seventh = seventh.toFloatArray(),
            ninth = ninth.toFloatArray(),
            eleventh = eleventh.toFloatArray(),
            thirteenth = thirteenth.toFloatArray(),
        )
    }

    private class FloatAccumulator(initialCapacity: Int = 4096) {
        private var values = FloatArray(initialCapacity.coerceAtLeast(1))
        var size: Int = 0
            private set

        fun append(input: FloatArray) {
            ensureCapacity(size + input.size)
            input.copyInto(values, destinationOffset = size)
            size += input.size
        }

        fun appendRows(source: FloatArray, width: Int, firstFrame: Int, endFrame: Int) {
            val start = firstFrame * width
            val end = endFrame * width
            val count = end - start
            ensureCapacity(size + count)
            source.copyInto(values, destinationOffset = size, startIndex = start, endIndex = end)
            size += count
        }

        fun toFloatArray(): FloatArray = values.copyOf(size)

        fun clear() {
            values = FloatArray(1)
            size = 0
        }

        private fun ensureCapacity(required: Int) {
            if (required <= values.size) return
            var capacity = values.size.coerceAtLeast(1)
            while (capacity < required) capacity *= 2
            values = values.copyOf(capacity)
        }
    }

    private class LongAccumulator(initialCapacity: Int = 1024) {
        private var values = LongArray(initialCapacity.coerceAtLeast(1))
        private var size = 0

        fun add(value: Long) {
            if (size == values.size) values = values.copyOf(values.size * 2)
            values[size++] = value
        }

        fun toLongArray(): LongArray = values.copyOf(size)
    }
}

private fun List<Double>.averageOrZero(): Double = if (isEmpty()) 0.0 else average()
