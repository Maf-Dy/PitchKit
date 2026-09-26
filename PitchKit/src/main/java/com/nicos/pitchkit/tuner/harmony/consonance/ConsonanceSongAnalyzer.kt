package com.nicos.pitchkit.tuner.harmony.consonance

import com.nicos.pitchkit.tuner.harmony.chordnet.CqtCpuFrontend
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtPlanDecoder
import com.nicos.pitchkit.tuner.harmony.chordnet.StreamingPcmResampler
import com.nicos.pitchkit.tuner.harmony.song.SongChordSegment
import com.nicos.pitchkit.tuner.harmony.song.SongHarmonyAnalysis
import com.nicos.pitchkit.tuner.harmony.song.SongSectionDetector
import java.security.MessageDigest
import kotlin.math.max

/** Offline Android port of the decomposed Conformer ACE model. */
class ConsonanceSongAnalyzer internal constructor(
    modelBytes: ByteArray,
    metadata: ConsonanceMetadata,
    cqtPlanBytes: ByteArray,
    dictionaryBytes: ByteArray,
    referenceA4Hz: Double = 440.0,
    private val preferFlats: Boolean = false,
    /** Null keeps the likelihood the dictionary asset declares. */
    likelihood: ConsonanceLikelihood? = null,
    /** Null keeps the chord-change penalty the dictionary asset declares. */
    transitionPenalty: Double? = null,
) : AutoCloseable {
    private val runner = ConsonanceOnnxRunner(modelBytes, metadata.modelSha256)
    private val frontend: CqtCpuFrontend
    private val dictionary: ConsonanceDictionary
    private val decoder: ConsonanceDictionaryDecoder

    /** The dictionary-decoder variant this analyzer actually runs. */
    val likelihood: ConsonanceLikelihood
    val transitionPenalty: Double

    /** e.g. `Consonance dictionary · competitive · p100`; recorded as the analysis backend. */
    val variantLabel: String
        get() = "Consonance dictionary · ${likelihood.id} · p${formatPenalty(transitionPenalty)}"
    private val resampler = StreamingPcmResampler(
        targetRate = ConsonanceContract.SAMPLE_RATE,
        pitchScale = 440.0 / referenceA4Hz,
    )
    private val audio = FloatAccumulator(ConsonanceContract.SAMPLE_RATE * 30)

    private var totalTargetSamples = 0L
    private var closed = false
    private var finished = false
    private var cachedResult: SongHarmonyAnalysis? = null

    init {
        require(referenceA4Hz in 300.0..600.0)
        require(modelBytes.size.toLong() == metadata.modelSize) {
            "Unexpected Consonance model size: ${modelBytes.size}"
        }
        val planSha = sha256(cqtPlanBytes)
        require(planSha == ConsonanceContract.PLAN_SHA256) {
            "Unexpected Consonance CQT plan SHA-256: $planSha"
        }
        val plan = CqtPlanDecoder.decodeAndVerify(cqtPlanBytes)
        require(plan.config.sampleRate.toInt() == ConsonanceContract.SAMPLE_RATE)
        require(plan.config.hopLength == ConsonanceContract.HOP_LENGTH)
        require(plan.config.nBins == ConsonanceContract.INPUT_BINS)
        require(plan.config.binsPerOctave == ConsonanceContract.BINS_PER_OCTAVE)
        require(!plan.config.logMagnitude) { "Consonance requires linear-magnitude CQT" }
        frontend = CqtCpuFrontend(plan)

        val dictionarySha = sha256(dictionaryBytes)
        require(dictionarySha == ConsonanceContract.DICTIONARY_SHA256) {
            "Unexpected Consonance dictionary SHA-256: $dictionarySha"
        }
        dictionary = ConsonanceDictionaryParser.parse(
            dictionaryBytes.toString(Charsets.UTF_8),
            preferFlats,
        )
        this.likelihood = likelihood ?: dictionary.likelihood
        this.transitionPenalty = transitionPenalty ?: dictionary.transitionPenalty
        require(this.transitionPenalty >= 0.0) {
            "Consonance transition penalty must not be negative: ${this.transitionPenalty}"
        }
        decoder = ConsonanceDictionaryDecoder(
            dictionary = dictionary,
            transitionPenalty = this.transitionPenalty,
            likelihood = this.likelihood,
        )
    }

    @Synchronized
    fun accept(samples: FloatArray, sampleRate: Int) {
        check(!closed) { "ConsonanceSongAnalyzer is closed" }
        check(!finished) { "ConsonanceSongAnalyzer has already finished" }
        if (samples.isEmpty()) return
        val target = resampler.process(samples, sampleRate)
        if (target.isEmpty()) return
        totalTargetSamples += target.size
        audio.append(target)
    }

    @Synchronized
    fun finish(durationMs: Long = 0L): SongHarmonyAnalysis {
        check(!closed) { "ConsonanceSongAnalyzer is closed" }
        cachedResult?.let { return it }
        check(!finished) { "ConsonanceSongAnalyzer has already finished" }
        finished = true

        val inferredDuration = totalTargetSamples * 1000L / ConsonanceContract.SAMPLE_RATE
        val finalDuration = max(durationMs, inferredDuration).coerceAtLeast(0L)
        if (audio.size < ConsonanceContract.HOP_LENGTH) {
            return emptyAnalysis(finalDuration).also { cachedResult = it }
        }

        // Raw heads of every chunk are kept until the end: the dictionary Viterbi
        // decodes the whole song in one pass, exactly as the Python reference does.
        val retained = mutableListOf<RetainedChunk>()
        val chunkSamples = ConsonanceContract.SAMPLE_RATE * ConsonanceContract.CHUNK_SECONDS
        var chunkStartSample = 0
        while (chunkStartSample < audio.size) {
            val actualSamples = minOf(chunkSamples, audio.size - chunkStartSample)
            val chunk = FloatArray(chunkSamples)
            audio.copyInto(
                destination = chunk,
                sourceStart = chunkStartSample,
                sourceEnd = chunkStartSample + actualSamples,
            )
            normalizePeakInPlace(chunk, actualSamples)

            val features = frontend.transform(chunk)
            require(features.frameCount == ConsonanceContract.SEQUENCE_FRAMES) {
                "Consonance CQT produced ${features.frameCount} frames for a " +
                    "${ConsonanceContract.CHUNK_SECONDS}s chunk; expected " +
                    "${ConsonanceContract.SEQUENCE_FRAMES}. Regenerate Consonance assets/frontend together."
            }
            val featureMajor = transposeFrameMajorToFeatureMajor(
                features.values,
                features.frameCount,
                features.binCount,
            )
            val heads = runner.infer(featureMajor)
            val chunkStartMs = chunkStartSample * 1000L / ConsonanceContract.SAMPLE_RATE
            val actualEndMs = (chunkStartSample + actualSamples) * 1000L /
                ConsonanceContract.SAMPLE_RATE

            // Zero-padded tail frames of the last chunk are dropped before decoding.
            var usableFrames = 0
            while (usableFrames < heads.frames) {
                val frameMs = frameTimeMs(chunkStartMs, usableFrames)
                if (frameMs >= actualEndMs || frameMs >= finalDuration) break
                usableFrames++
            }
            if (usableFrames > 0) retained += RetainedChunk(heads, usableFrames, chunkStartMs)
            chunkStartSample += actualSamples
        }
        audio.clear()

        val chords = buildSegments(retained, finalDuration)
        return SongHarmonyAnalysis(
            durationMs = finalDuration,
            chords = chords,
            sections = SongSectionDetector.detect(chords, finalDuration),
        ).also { cachedResult = it }
    }

    private class RetainedChunk(
        val heads: ConsonanceHeads,
        val usableFrames: Int,
        val startMs: Long,
    )

    /**
     * Frame times follow the analyzer's own convention (chunk offset plus
     * frameIndex * HOP_LENGTH / SAMPLE_RATE in whole milliseconds) rather than the
     * reference script's CHUNK_SECONDS / CHUNK_FRAMES seconds; both describe the
     * same 512-sample hop grid.
     */
    private fun frameTimeMs(chunkStartMs: Long, frameIndex: Int): Long = chunkStartMs +
        frameIndex.toLong() * ConsonanceContract.HOP_LENGTH * 1000L / ConsonanceContract.SAMPLE_RATE

    private fun buildSegments(
        retained: List<RetainedChunk>,
        durationMs: Long,
    ): List<SongChordSegment> {
        val totalFrames = retained.sumOf { it.usableFrames }
        if (totalFrames <= 0) return emptyList()

        val heads = ConsonanceHeads(
            frames = totalFrames,
            root = FloatArray(totalFrames * ConsonanceContract.ROOT_COUNT),
            bass = FloatArray(totalFrames * ConsonanceContract.BASS_COUNT),
            pitch = FloatArray(totalFrames * ConsonanceContract.PITCH_COUNT),
        )
        val frameTimes = LongArray(totalFrames)
        var written = 0
        for (chunk in retained) {
            chunk.heads.root.copyInto(
                destination = heads.root,
                destinationOffset = written * ConsonanceContract.ROOT_COUNT,
                startIndex = 0,
                endIndex = chunk.usableFrames * ConsonanceContract.ROOT_COUNT,
            )
            chunk.heads.bass.copyInto(
                destination = heads.bass,
                destinationOffset = written * ConsonanceContract.BASS_COUNT,
                startIndex = 0,
                endIndex = chunk.usableFrames * ConsonanceContract.BASS_COUNT,
            )
            chunk.heads.pitch.copyInto(
                destination = heads.pitch,
                destinationOffset = written * ConsonanceContract.PITCH_COUNT,
                startIndex = 0,
                endIndex = chunk.usableFrames * ConsonanceContract.PITCH_COUNT,
            )
            for (frame in 0 until chunk.usableFrames) {
                frameTimes[written + frame] = frameTimeMs(chunk.startMs, frame)
            }
            written += chunk.usableFrames
        }

        val segments = decoder.decode(heads)
        val chords = mutableListOf<SongChordSegment>()
        for (segment in segments) {
            val label = segment.candidate.displayLabel ?: continue
            val startMs = frameTimes[segment.startFrame].coerceAtMost(durationMs)
            val endMs = (frameTimes.getOrNull(segment.endFrame) ?: durationMs)
                .coerceAtMost(durationMs)
            if (endMs <= startMs) continue
            chords += SongChordSegment(
                label = label,
                startMs = startMs,
                endMs = endMs,
                confidence = segment.confidence,
                root = segment.candidate.root.takeIf { it in 0..11 }?.let(::noteName),
                bass = segment.candidate.bass.takeIf { it in 0..11 }?.let(::noteName),
                pitchClasses = pitchClasses(segment.candidate.mask),
            )
        }
        return chords
    }

    private fun pitchClasses(mask: Int): List<String> = (0 until 12)
        .filter { mask and (1 shl it) != 0 }
        .map(::noteName)

    private fun transposeFrameMajorToFeatureMajor(
        values: FloatArray,
        frames: Int,
        bins: Int,
    ): FloatArray {
        require(values.size >= frames * bins)
        val output = FloatArray(frames * bins)
        for (frame in 0 until frames) {
            val source = frame * bins
            for (bin in 0 until bins) {
                output[bin * frames + frame] = values[source + bin]
            }
        }
        return output
    }

    private fun normalizePeakInPlace(values: FloatArray, actualSamples: Int) {
        var peak = 0f
        for (index in 0 until actualSamples.coerceAtMost(values.size)) {
            peak = maxOf(peak, kotlin.math.abs(values[index]))
        }
        if (peak <= 1e-8f) return
        for (index in 0 until actualSamples.coerceAtMost(values.size)) values[index] /= peak
    }

    private val sharpNames = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    private val flatNames = arrayOf("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
    private fun noteName(pc: Int): String = if (preferFlats) flatNames[pc] else sharpNames[pc]

    /** Whole penalties read as `p100`; anything else keeps its decimals. */
    private fun formatPenalty(value: Double): String =
        if (value == value.toLong().toDouble()) value.toLong().toString() else value.toString()

    private fun emptyAnalysis(durationMs: Long): SongHarmonyAnalysis = SongHarmonyAnalysis(
        durationMs = durationMs,
        chords = emptyList(),
        sections = emptyList(),
    )

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        runner.close()
        audio.clear()
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private class FloatAccumulator(initialCapacity: Int) {
        private var values = FloatArray(initialCapacity.coerceAtLeast(1))
        var size: Int = 0
            private set

        fun append(input: FloatArray) {
            ensureCapacity(size + input.size)
            input.copyInto(values, destinationOffset = size)
            size += input.size
        }

        fun copyInto(
            destination: FloatArray,
            sourceStart: Int,
            sourceEnd: Int,
        ) {
            values.copyInto(
                destination = destination,
                destinationOffset = 0,
                startIndex = sourceStart,
                endIndex = sourceEnd,
            )
        }

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
}
