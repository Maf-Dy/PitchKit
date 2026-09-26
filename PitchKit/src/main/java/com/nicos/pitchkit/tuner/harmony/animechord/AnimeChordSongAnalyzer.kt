package com.nicos.pitchkit.tuner.harmony.animechord

import com.nicos.pitchkit.tuner.harmony.chordnet.StreamingPcmResampler
import com.nicos.pitchkit.tuner.harmony.song.SongChordSegment
import com.nicos.pitchkit.tuner.harmony.song.SongHarmonyAnalysis
import com.nicos.pitchkit.tuner.harmony.song.SongSection
import com.nicos.pitchkit.tuner.harmony.song.SongSectionDetector
import kotlin.math.max
import kotlin.math.roundToLong

/**
 * Offline `anime Chord-Transcription` song analyzer — the richest chord vocabulary this
 * app ships, and the most expensive.
 *
 * The authority chain is upstream's end to end: `RecursiveCQT` + whole-clip
 * standardisation, the epoch-150 backbone as an opset-17 ONNX graph, the author's sticky
 * HMM over the stitched per-frame logits, the author's event builder, and the author's own
 * 62-quality label vocabulary. Only the windowing is ours, and it is the windowing the PC
 * lane `anime-hmm-w30-o50` measured: 30 s windows at 50 % overlap, centre commit, window 0
 * unprimed.
 *
 * Costs, measured in `.accuracy-work/annotations/anime-windowed-report.md` §7 and
 * projected to one ARM big core: **~31 CPU-seconds per audio-minute** (about 64 wall
 * seconds for a four-minute song at four threads) and **~455 MB of ONNX Runtime arena**,
 * which is comfortable on an 8 GB device and uncomfortable on a 4 GB one. Everything this
 * class does with its own memory is arranged around that: the feature matrix is one
 * channel (the app decodes mono, and the graph's stereo input is that channel duplicated),
 * the 745-wide logits are consumed window by window instead of being stitched into a
 * whole-song matrix, and the Viterbi keeps one 745-bit stay mask per frame — 4 KB per
 * audio-second where the logits would be 128 KB.
 */
class AnimeChordSongAnalyzer internal constructor(
    modelBytes: ByteArray,
    planBytes: ByteArray,
    vocabularyJson: String,
    referenceA4Hz: Double = 440.0,
    preferFlats: Boolean = true,
    threads: Int = AnimeChordOnnxRunner.defaultThreads(),
) : AutoCloseable {
    private val frontend = AnimeChordCqtFrontend(planBytes)
    private val runner = AnimeChordOnnxRunner(modelBytes, threads)
    private val vocabulary = AnimeChordVocabulary.parse(vocabularyJson)
    private val resampler = StreamingPcmResampler(
        targetRate = AnimeChordContract.SAMPLE_RATE,
        pitchScale = 440.0 / referenceA4Hz,
    )
    private val pcm = FloatAccumulator(initialCapacity = AnimeChordContract.SAMPLE_RATE * 30)

    /**
     * The model's own root names are flat (`Db`, `Eb`, `Gb`, `Ab`, `Bb`) and the Harte
     * shorthand is built from them, so a sharp-preferring caller only changes how the
     * root and bass columns of a stored segment read.
     */
    private val preferFlatNames = preferFlats

    private var totalTargetSamples = 0L
    private var closed = false
    private var finished = false
    private var cachedResult: SongHarmonyAnalysis? = null

    val backend: String = AnimeChordContract.BACKEND

    init {
        require(referenceA4Hz in 300.0..600.0)
        require(AnimeChordContract.OVERLAP_FRAMES < AnimeChordContract.WINDOW_FRAMES)
        require(vocabulary.rootChordCount == AnimeChordContract.ROOT_CHORD_COUNT)
    }

    /** Feed decoded mono PCM values in the -1..1 range. */
    @Synchronized
    fun accept(samples: FloatArray, sampleRate: Int) {
        check(!closed) { "AnimeChordSongAnalyzer is closed" }
        check(!finished) { "AnimeChordSongAnalyzer has already finished" }
        if (samples.isEmpty()) return

        val target = resampler.process(samples, sampleRate)
        if (target.isEmpty()) return
        totalTargetSamples += target.size
        pcm.append(target)
    }

    @Synchronized
    fun finish(durationMs: Long = 0L): SongHarmonyAnalysis {
        check(!closed) { "AnimeChordSongAnalyzer is closed" }
        cachedResult?.let { return it }
        check(!finished) { "AnimeChordSongAnalyzer has already finished" }
        finished = true

        val inferredDuration = totalTargetSamples * 1000L / AnimeChordContract.SAMPLE_RATE
        val finalDuration = max(durationMs, inferredDuration)
        if (pcm.size < AnimeChordContract.CROP_N_FFT) {
            return emptyAnalysis(finalDuration).also { cachedResult = it }
        }

        val features = frontend.transform(pcm.toFloatArray())
        pcm.clear()
        val decodedFrames = features.cropLength
        if (features.frameCount <= 0 || decodedFrames <= 1) {
            return emptyAnalysis(finalDuration).also { cachedResult = it }
        }

        val root = AnimeChordViterbi(
            classCount = AnimeChordContract.ROOT_CHORD_COUNT,
            expectedFrames = decodedFrames,
        )
        val bass = AnimeChordViterbi(
            classCount = AnimeChordContract.BASS_COUNT,
            expectedFrames = decodedFrames,
        )
        val bins = AnimeChordContract.INPUT_BINS
        var appended = 0

        for (window in AnimeChordWindowPlan.windows(features.frameCount)) {
            // Frames past `crop_length` are the tail the reference trims before decoding.
            val commitEnd = window.commitEnd
                .coerceAtMost(decodedFrames - window.startFrame)
                .coerceAtLeast(window.commitStart)
            if (commitEnd <= window.commitStart) {
                if (window.startFrame >= decodedFrames) break
                continue
            }
            val slice = features.values.copyOfRange(
                window.startFrame * bins,
                (window.startFrame + window.usableFrames) * bins,
            )
            val heads = runner.infer(
                features = slice,
                frames = window.usableFrames,
                commitStart = window.commitStart,
                commitEnd = commitEnd,
            )
            for (local in 0 until heads.frames) {
                root.add(heads.rootChord, local * AnimeChordContract.ROOT_CHORD_COUNT)
                bass.add(heads.bass, local * AnimeChordContract.BASS_COUNT)
            }
            appended += heads.frames
        }

        check(appended == decodedFrames) {
            "anime window plan committed $appended of $decodedFrames frames"
        }

        val events = AnimeChordDecoder.events(
            rootPath = root.decode(),
            bassPath = bass.decode(),
            vocabulary = vocabulary,
        )
        val chords = buildChordSegments(events, finalDuration)
        return SongHarmonyAnalysis(
            durationMs = finalDuration,
            chords = chords,
            sections = SongSectionDetector.detect(chords, finalDuration),
        ).also { cachedResult = it }
    }

    private fun buildChordSegments(
        events: List<AnimeChordEvent>,
        durationMs: Long,
    ): List<SongChordSegment> = events.mapNotNull { event ->
        val label = vocabulary.harteLabel(event.rootChordIndex, event.bassIndex)
            ?: return@mapNotNull null
        val startMs = (event.startSeconds * 1000.0).roundToLong().coerceIn(0L, durationMs)
        val endMs = (event.endSeconds * 1000.0).roundToLong().coerceIn(0L, durationMs)
        if (endMs <= startMs) return@mapNotNull null
        val rootPc = vocabulary.rootPitchClass(event.rootChordIndex)
        val bassPc = vocabulary.bassPitchClass(event.rootChordIndex, event.bassIndex)
        SongChordSegment(
            label = label,
            startMs = startMs,
            endMs = endMs,
            // The decode is a maximum-likelihood path, not a per-frame posterior, and the
            // reference publishes no per-event score; reporting one would invent it.
            confidence = 1.0,
            root = rootPc?.let(::noteName),
            bass = (bassPc ?: rootPc)?.let(::noteName),
            pitchClasses = vocabulary.pitchClassNames(event.rootChordIndex, event.bassIndex)
                .map(::respell),
        )
    }

    private fun noteName(pitchClass: Int): String = respell(vocabulary.noteName(pitchClass))

    private fun respell(name: String): String =
        if (preferFlatNames) name else SHARP_SPELLINGS[name] ?: name

    private fun emptyAnalysis(durationMs: Long): SongHarmonyAnalysis = SongHarmonyAnalysis(
        durationMs = durationMs,
        chords = emptyList(),
        sections = if (durationMs > 0L) listOf(SongSection("A", 0L, durationMs)) else emptyList(),
    )

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        runCatching { runner.close() }
        pcm.clear()
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

    private companion object {
        val SHARP_SPELLINGS = mapOf(
            "Db" to "C#", "Eb" to "D#", "Gb" to "F#", "Ab" to "G#", "Bb" to "A#",
        )
    }
}
