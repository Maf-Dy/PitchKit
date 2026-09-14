package com.nicos.pitchkit.tuner.harmony.lvchordia

import android.os.SystemClock
import android.util.Log
import com.nicos.pitchkit.BuildConfig
import com.nicos.pitchkit.tuner.harmony.PitchClassChordReranker
import com.nicos.pitchkit.tuner.harmony.chordnet.StreamingPcmResampler
import com.nicos.pitchkit.tuner.harmony.song.SongChordSegment
import com.nicos.pitchkit.tuner.harmony.song.SongHarmonyAnalysis
import com.nicos.pitchkit.tuner.harmony.song.SongSection
import com.nicos.pitchkit.tuner.harmony.song.SongSectionDetector
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

/** Offline LV Song large-vocabulary analyzer using the ensemble + dictionary HMM as authority. */
class LvChordiaSongAnalyzer internal constructor(
    modelBytes: List<ByteArray>,
    dictionaryJson: String,
    lowPlanBytes: ByteArray,
    highPlanBytes: ByteArray,
    referenceA4Hz: Double = 440.0,
    preferFlats: Boolean = false,
) : AutoCloseable {
    private companion object {
        const val MODEL_WINDOW_FRAMES = 512
        const val MODEL_OVERLAP_FRAMES = 64
        const val MODEL_HALF_OVERLAP = MODEL_OVERLAP_FRAMES / 2
        const val MODEL_STEP_FRAMES = MODEL_WINDOW_FRAMES - MODEL_OVERLAP_FRAMES
        const val HARMONY_PRESENCE_THRESHOLD = 0.28f
        const val HARMONY_MEAN_WEIGHT = 0.65
        const val HARMONY_PERSISTENCE_WEIGHT = 0.35
    }

    private data class DiagnosticSegment(
        val start: Int,
        val endExclusive: Int,
        val label: String?,
    )

    private val frontend = LvChordiaHybridCqtFrontend(lowPlanBytes, highPlanBytes)
    private val runners = modelBytes.map(::LvChordiaOnnxRunner)
    private val inferencePool = Executors.newFixedThreadPool(minOf(3, runners.size))
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
    }

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

        val startedAt = SystemClock.elapsedRealtimeNanos()
        val features = frontend.transform(pcm.toFloatArray())
        val cqtDoneAt = SystemClock.elapsedRealtimeNanos()
        pcm.clear()
        if (features.frameCount <= 0) {
            return emptyAnalysis(finalDuration).also { cachedResult = it }
        }

        val committed = HeadsAccumulator()
        val frameNumbers = LongAccumulator(initialCapacity = features.frameCount)
        analyzeFeatureWindows(features, committed, frameNumbers)
        val ensembleDoneAt = SystemClock.elapsedRealtimeNanos()

        val heads = committed.build()
        if (heads.frames <= 0) {
            return emptyAnalysis(finalDuration).also { cachedResult = it }
        }
        val frames = frameNumbers.toLongArray()
        require(frames.size == heads.frames)

        val modelDecoded = sequenceDecoder.decode(heads)
        val hmmDoneAt = SystemClock.elapsedRealtimeNanos()
        require(modelDecoded.size == frames.size)
        logHarmonyDiagnostics(modelDecoded, frames, heads)

        // Upstream LV-Chordia's full-song path is model ensemble -> dictionary
        // HMM -> chord segments. Do not silently rewrite that decoded sequence
        // from a second handcrafted chroma scorer; doing so can turn a correct,
        // low-confidence model result into a confident but different chord.
        val decoded = modelDecoded

        val chords = buildChordSegments(decoded, frames, finalDuration)
        val result = SongHarmonyAnalysis(
            durationMs = finalDuration,
            chords = chords,
            sections = SongSectionDetector.detect(chords, finalDuration),
        )
        logTiming(
            startedAt = startedAt,
            cqtDoneAt = cqtDoneAt,
            ensembleDoneAt = ensembleDoneAt,
            hmmDoneAt = hmmDoneAt,
            frames = features.frameCount,
            chordCount = chords.size,
        )
        cachedResult = result
        return result
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

            val futures = runners.map { runner ->
                inferencePool.submit<LvChordiaHeads> {
                    runner.infer(windowValues, frameCount)
                }
            }
            val average = LvChordiaHeads.average(futures.map { it.get() })

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

    /**
     * Retained for diagnostics/experiments, but deliberately not part of the
     * production LV Song path. The HMM/dictionary result above is authoritative.
     */
    private fun rerankExtensionsWithPitchEvidence(
        decoded: List<LvChordiaDecodedFrame>,
        frames: LongArray,
        harmonyChroma: FloatArray,
        bassChroma: FloatArray,
        chromaFrameCount: Int,
    ): List<LvChordiaDecodedFrame> {
        if (
            decoded.isEmpty() ||
            chromaFrameCount <= 0 ||
            harmonyChroma.size < chromaFrameCount * 12 ||
            bassChroma.size < chromaFrameCount * 12
        ) {
            return decoded
        }

        val result = decoded.toMutableList()
        var start = 0
        while (start < decoded.size) {
            val label = decoded[start].label
            var end = start + 1
            while (end < decoded.size && decoded[end].label == label) end++

            if (label != null) {
                val firstSourceFrame = frames[start]
                    .coerceIn(0L, (chromaFrameCount - 1).toLong())
                    .toInt()
                val lastSourceFrame = frames[end - 1]
                    .coerceIn(firstSourceFrame.toLong(), (chromaFrameCount - 1).toLong())
                    .toInt()
                val frameRange = firstSourceFrame..lastSourceFrame
                val harmonyEvidence = persistentHarmonyEvidence(
                    chroma = harmonyChroma,
                    frameCount = chromaFrameCount,
                    frameIndices = frameRange,
                )
                val bassEvidence = PitchClassChordReranker.averageEvidence(
                    chroma = bassChroma,
                    frameCount = chromaFrameCount,
                    frameIndices = frameRange,
                )

                val reranked = PitchClassChordReranker.rerank(label, harmonyEvidence)
                val rootResolved = PitchClassChordReranker.resolveEquivalentRoot(
                    label = reranked.label,
                    pitchEvidence = harmonyEvidence,
                    bassEvidence = bassEvidence,
                )
                val chosen = if (rootResolved.changed) rootResolved else reranked

                if (chosen.changed) {
                    for (index in start until end) {
                        result[index] = decoded[index].copy(label = chosen.label)
                    }
                    if (BuildConfig.DEBUG) {
                        Log.d(
                            "PitchKitHarmony",
                            "LV Song correction $label -> ${chosen.label} " +
                                "frames=$firstSourceFrame-$lastSourceFrame " +
                                "score=${"%.3f".format(chosen.originalScore)}->${"%.3f".format(chosen.score)} " +
                                "harmony=[${pitchSummary(harmonyEvidence)}] " +
                                "bass=[${pitchSummary(bassEvidence)}]",
                        )
                    }
                }
            }
            start = end
        }
        return result
    }

    private fun persistentHarmonyEvidence(
        chroma: FloatArray,
        frameCount: Int,
        frameIndices: IntRange,
    ): FloatArray {
        if (frameCount <= 0 || chroma.size < frameCount * 12) return FloatArray(12)
        val first = frameIndices.first.coerceIn(0, frameCount - 1)
        val last = frameIndices.last.coerceIn(first, frameCount - 1)
        val count = last - first + 1
        if (count <= 3) {
            return PitchClassChordReranker.averageEvidence(chroma, frameCount, first..last)
        }

        val mean = DoubleArray(12)
        val present = IntArray(12)
        for (frame in first..last) {
            val offset = frame * 12
            for (pc in 0 until 12) {
                val value = chroma[offset + pc].coerceIn(0f, 1f)
                mean[pc] += value
                if (value >= HARMONY_PRESENCE_THRESHOLD) present[pc]++
            }
        }

        val combined = DoubleArray(12)
        for (pc in 0 until 12) {
            val average = mean[pc] / count.toDouble()
            val persistence = present[pc].toDouble() / count.toDouble()
            combined[pc] = HARMONY_MEAN_WEIGHT * average +
                HARMONY_PERSISTENCE_WEIGHT * persistence
        }

        val peak = combined.maxOrNull()?.coerceAtLeast(0.0) ?: 0.0
        if (peak <= 1e-12) return FloatArray(12)
        return FloatArray(12) { pc -> (combined[pc] / peak).coerceIn(0.0, 1.0).toFloat() }
    }

    private fun pitchSummary(evidence: FloatArray): String {
        val names = arrayOf("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
        return evidence.indices
            .sortedByDescending { evidence[it] }
            .take(6)
            .joinToString(",") { index -> "${names[index]}=${"%.2f".format(evidence[index])}" }
    }

    private fun logHarmonyDiagnostics(
        decoded: List<LvChordiaDecodedFrame>,
        frames: LongArray,
        heads: LvChordiaHeads,
    ) {
        if (!BuildConfig.DEBUG || decoded.isEmpty()) return

        val segments = mutableListOf<DiagnosticSegment>()
        var start = 0
        while (start < decoded.size) {
            val label = decoded[start].label
            var end = start + 1
            while (end < decoded.size && decoded[end].label == label) end++
            segments += DiagnosticSegment(start, end, label)
            start = end
        }

        val midpoints = IntArray(segments.size) { index ->
            val segment = segments[index]
            (segment.start + segment.endExclusive - 1) / 2
        }
        val diagnostics = sequenceDecoder
            .diagnoseFrames(heads, midpoints, limit = 3)
            .associateBy { it.frame }

        for ((index, segment) in segments.withIndex()) {
            val midpoint = midpoints[index]
            val diagnostic = diagnostics[midpoint] ?: continue
            val rawTop = diagnostic.topCandidates.joinToString(separator = " | ") { candidate ->
                "${candidate.displayLabel ?: candidate.rawLabel}=${"%.3f".format(candidate.confidence)}"
            }
            val startMs = frameToMs(frames[segment.start])
            val endFrameIndex = (segment.endExclusive - 1).coerceAtMost(frames.lastIndex)
            val endMs = frameToMs(frames[endFrameIndex])
            Log.d(
                "PitchKitHarmony",
                "LV Song ${startMs}-${endMs}ms hmm=${segment.label ?: "N"} " +
                    "rawTop=[$rawTop] ${diagnostic.headSummary}",
            )
        }
    }

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
                        confidence = decoded.subList(start, end).map { it.confidence }.averageOrZero(),
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

    private fun logTiming(
        startedAt: Long,
        cqtDoneAt: Long,
        ensembleDoneAt: Long,
        hmmDoneAt: Long,
        frames: Int,
        chordCount: Int,
    ) {
        if (!BuildConfig.DEBUG) return
        val cqtMs = (cqtDoneAt - startedAt) / 1_000_000.0
        val ensembleMs = (ensembleDoneAt - cqtDoneAt) / 1_000_000.0
        val hmmMs = (hmmDoneAt - ensembleDoneAt) / 1_000_000.0
        val totalMs = (hmmDoneAt - startedAt) / 1_000_000.0
        Log.d(
            "PitchKitPerf",
            "LV Song frames=$frames chords=$chordCount cqt=${"%.1f".format(cqtMs)}ms " +
                "ensemble=${"%.1f".format(ensembleMs)}ms hmm=${"%.1f".format(hmmMs)}ms " +
                "total=${"%.1f".format(totalMs)}ms",
        )
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        inferencePool.shutdownNow()
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
