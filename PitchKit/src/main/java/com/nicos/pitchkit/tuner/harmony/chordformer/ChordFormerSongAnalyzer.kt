package com.nicos.pitchkit.tuner.harmony.chordformer

import com.nicos.pitchkit.tuner.harmony.chordnet.StreamingPcmResampler
import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaDecodedFrame
import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaDictionaryDecoder
import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaDictionaryParser
import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaHeads
import com.nicos.pitchkit.tuner.harmony.song.SongChordSegment
import com.nicos.pitchkit.tuner.harmony.song.SongHarmonyAnalysis
import com.nicos.pitchkit.tuner.harmony.song.SongSection
import com.nicos.pitchkit.tuner.harmony.song.SongSectionDetector
import kotlin.math.max

/**
 * Offline ChordFormer song analyzer (fold 3, `submission` vocabulary).
 *
 * The authority chain is upstream's: CQTV2 dB feature -> conformer heads -> dictionary
 * XHMM -> segments. Only the sequence length is ours. Upstream runs the conformer over a
 * whole song in one call; `nn.MultiheadAttention` bakes the traced length into the
 * exported graph, so the model runs on fixed 2048-frame (~47.6 s) windows with a
 * 512-frame overlap and only the well-contextualised centre of each window is committed.
 * The conformer uses full global attention with no positional encoding, so a window is a
 * legitimate unit of inference; the feasibility run measured 2048/512 at 0.6 root /
 * 1.4 family below the unwindowed whole-song reference on the owner's corpus.
 *
 * The feature matrix is normalised once, over the whole song, before any windowing: the
 * dB reference is a global `np.max` and must not be recomputed per window.
 *
 * One deviation from that reference run, at the very end: the final window is padded out
 * to 2048 frames with the -80 dB floor, the value true silence carries under `ref=np.max`,
 * and the padded frames are dropped before decoding. Upstream never pads, so real frames in
 * that window can attend to it. (The phone diagnosis did not emulate that deviation, so it
 * is unmeasured.)
 *
 * The first window used to be primed -- started half an overlap early with the opening of
 * the song reflected into the lead-in. `chordformer-phone-diagnosis.md` measured that at
 * -5.2 root / -4.8 family against this plan, because the priming buys synthetic left
 * context by giving up 256 frames of real right context in the one window that has to read
 * an intro. It is gone: window 0 starts at frame 0, reads 2048 real frames and commits from
 * its own frame 0.
 */
class ChordFormerSongAnalyzer internal constructor(
    modelBytes: ByteArray,
    dictionaryJson: String,
    lowPlanBytes: ByteArray,
    highPlanBytes: ByteArray,
    referenceA4Hz: Double = 440.0,
    preferFlats: Boolean = false,
) : AutoCloseable {
    private val frontend = ChordFormerCqtFrontend(lowPlanBytes, highPlanBytes)
    private val runner = ChordFormerOnnxRunner(modelBytes)
    private val dictionary = LvChordiaDictionaryParser.parse(dictionaryJson, preferFlats)
    private val sequenceDecoder = LvChordiaDictionaryDecoder(dictionary)
    private val resampler = StreamingPcmResampler(
        targetRate = ChordFormerContract.SAMPLE_RATE,
        pitchScale = 440.0 / referenceA4Hz,
    )
    private val pcm = FloatAccumulator(initialCapacity = ChordFormerContract.SAMPLE_RATE * 30)

    private var totalTargetSamples = 0L
    private var closed = false
    private var finished = false
    private var cachedResult: SongHarmonyAnalysis? = null

    val backend: String = ChordFormerContract.BACKEND

    init {
        require(referenceA4Hz in 300.0..600.0)
        require(ChordFormerContract.OVERLAP_FRAMES < ChordFormerContract.WINDOW_FRAMES)
        require(ChordFormerContract.HALF_OVERLAP_FRAMES * 2 == ChordFormerContract.OVERLAP_FRAMES)
        require(dictionary.transitionPenalty == ChordFormerContract.TRANSITION_PENALTY) {
            "ChordFormer dictionary declares penalty ${dictionary.transitionPenalty}"
        }
    }

    /** Feed decoded mono PCM values in the -1..1 range. */
    @Synchronized
    fun accept(samples: FloatArray, sampleRate: Int) {
        check(!closed) { "ChordFormerSongAnalyzer is closed" }
        check(!finished) { "ChordFormerSongAnalyzer has already finished" }
        if (samples.isEmpty()) return

        val target = resampler.process(samples, sampleRate)
        if (target.isEmpty()) return
        totalTargetSamples += target.size
        pcm.append(target)
    }

    @Synchronized
    fun finish(durationMs: Long = 0L): SongHarmonyAnalysis {
        check(!closed) { "ChordFormerSongAnalyzer is closed" }
        cachedResult?.let { return it }
        check(!finished) { "ChordFormerSongAnalyzer has already finished" }
        finished = true

        val inferredDuration = totalTargetSamples * 1000L / ChordFormerContract.SAMPLE_RATE
        val finalDuration = max(durationMs, inferredDuration)
        if (pcm.size < ChordFormerContract.HOP_LENGTH) {
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

        val decoded = sequenceDecoder.decode(heads)
        require(decoded.size == frames.size)

        val chords = buildChordSegments(decoded, frames, finalDuration)
        return SongHarmonyAnalysis(
            durationMs = finalDuration,
            chords = chords,
            sections = SongSectionDetector.detect(chords, finalDuration),
        ).also { cachedResult = it }
    }

    /**
     * Fixed 2048-frame windows, half-overlap commit, over the already-normalised
     * whole-song feature matrix.
     *
     * Window 0 starts at real frame 0 and commits from its own frame 0; every later window
     * commits only its centre. Each real frame is committed exactly once, by the window
     * that saw the most context around it.
     */
    private fun analyzeFeatureWindows(
        features: ChordFormerFeatures,
        committed: HeadsAccumulator,
        frameNumbers: LongAccumulator,
    ) {
        val bins = ChordFormerContract.INPUT_BINS
        val buffer = FloatArray(ChordFormerContract.WINDOW_FRAMES * bins)

        for (window in ChordFormerWindowPlan.windows(features.frameCount)) {
            // The graph is fixed at 2048 frames, so a short window is padded with the dB
            // floor -- the value real silence carries under ref=np.max -- and the padded
            // frames are dropped afterwards.
            java.util.Arrays.fill(buffer, ChordFormerContract.SILENCE_DB)
            for (local in 0 until window.usableFrames) {
                val source = ChordFormerWindowPlan.sourceFrame(
                    window.startFrame + local,
                    features.frameCount,
                )
                features.values.copyInto(
                    destination = buffer,
                    destinationOffset = local * bins,
                    startIndex = source * bins,
                    endIndex = (source + 1) * bins,
                )
            }

            val heads = runner.infer(buffer)

            if (window.commitStart < window.commitEnd) {
                committed.append(heads, window.commitStart, window.commitEnd)
                for (local in window.commitStart until window.commitEnd) {
                    frameNumbers.add((window.startFrame + local).toLong())
                }
            }
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
                            .let { if (it.isEmpty()) 0.0 else it.average() },
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
                result[result.lastIndex] = previous.copy(endMs = segment.endMs, confidence = confidence)
            } else {
                result += segment
            }
        }
        return result
    }

    private fun frameToMs(frame: Long): Long =
        frame * ChordFormerContract.HOP_LENGTH * 1000L / ChordFormerContract.SAMPLE_RATE

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        runCatching { runner.close() }
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
            triad.appendRows(heads.triad, ChordFormerContract.TRIAD_COUNT, firstFrame, endFrame)
            bass.appendRows(heads.bass, ChordFormerContract.BASS_COUNT, firstFrame, endFrame)
            seventh.appendRows(heads.seventh, ChordFormerContract.SEVENTH_COUNT, firstFrame, endFrame)
            ninth.appendRows(heads.ninth, ChordFormerContract.NINTH_COUNT, firstFrame, endFrame)
            eleventh.appendRows(heads.eleventh, ChordFormerContract.ELEVENTH_COUNT, firstFrame, endFrame)
            thirteenth.appendRows(heads.thirteenth, ChordFormerContract.THIRTEENTH_COUNT, firstFrame, endFrame)
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
            ensureCapacity(size + (end - start))
            source.copyInto(values, destinationOffset = size, startIndex = start, endIndex = end)
            size += end - start
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
