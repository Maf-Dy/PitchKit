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
    referenceA4Hz: Double = 440.0,
    private val preferFlats: Boolean = false,
) : AutoCloseable {
    private val runner = ConsonanceOnnxRunner(modelBytes, metadata.modelSha256)
    private val frontend: CqtCpuFrontend
    private val decoder = ConsonanceChordDecoder(preferFlats = preferFlats)
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

        val frames = mutableListOf<TimedFrame>()
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
            val decoded = decoder.decode(heads)
            val chunkStartMs = chunkStartSample * 1000L / ConsonanceContract.SAMPLE_RATE
            val actualEndMs = (chunkStartSample + actualSamples) * 1000L /
                ConsonanceContract.SAMPLE_RATE

            for (frameIndex in decoded.indices) {
                val frameMs = chunkStartMs +
                    frameIndex.toLong() * ConsonanceContract.HOP_LENGTH * 1000L /
                    ConsonanceContract.SAMPLE_RATE
                if (frameMs >= actualEndMs || frameMs >= finalDuration) break
                frames += TimedFrame(frameMs, decoded[frameIndex])
            }
            chunkStartSample += actualSamples
        }
        audio.clear()

        val chords = buildSegments(frames, finalDuration)
        return SongHarmonyAnalysis(
            durationMs = finalDuration,
            chords = chords,
            sections = SongSectionDetector.detect(chords, finalDuration),
        ).also { cachedResult = it }
    }

    private data class TimedFrame(
        val timeMs: Long,
        val chord: ConsonanceDecodedFrame,
    )

    private fun buildSegments(
        frames: List<TimedFrame>,
        durationMs: Long,
    ): List<SongChordSegment> {
        if (frames.isEmpty()) return emptyList()
        val raw = mutableListOf<SongChordSegment>()
        var start = 0
        while (start < frames.size) {
            val label = frames[start].chord.label
            var end = start + 1
            while (end < frames.size && frames[end].chord.label == label) end++

            if (label != null) {
                val slice = frames.subList(start, end)
                val startMs = slice.first().timeMs.coerceAtMost(durationMs)
                val endMs = (frames.getOrNull(end)?.timeMs ?: durationMs).coerceAtMost(durationMs)
                if (endMs > startMs) {
                    raw += SongChordSegment(
                        label = label,
                        startMs = startMs,
                        endMs = endMs,
                        confidence = slice.map { it.chord.confidence }.averageOrZero(),
                        root = modeString(slice.mapNotNull { it.chord.root }),
                        bass = modeString(slice.mapNotNull { it.chord.bass }),
                        pitchClasses = persistentPitchClasses(slice),
                    )
                }
            }
            start = end
        }
        return removeShortSegments(raw)
    }

    private fun persistentPitchClasses(frames: List<TimedFrame>): List<String> {
        if (frames.isEmpty()) return emptyList()
        val average = DoubleArray(12)
        for (frame in frames) {
            for (pc in 0 until 12) average[pc] += frame.chord.pitchProbabilities[pc]
        }
        for (pc in 0 until 12) average[pc] /= frames.size.toDouble()
        return average.indices
            .filter { average[it] >= 0.45 }
            .sortedByDescending { average[it] }
            .map(::noteName)
    }

    private fun removeShortSegments(input: List<SongChordSegment>): List<SongChordSegment> {
        if (input.isEmpty()) return input
        val minimumMs = (ConsonanceContract.MIN_SEGMENT_SECONDS * 1000.0).toLong()
        val output = mutableListOf<SongChordSegment>()
        var index = 0
        while (index < input.size) {
            val segment = input[index]
            val duration = segment.endMs - segment.startMs
            if (duration < minimumMs) {
                if (output.isNotEmpty()) {
                    val previous = output.removeAt(output.lastIndex)
                    output += previous.copy(endMs = segment.endMs)
                } else if (index + 1 < input.size) {
                    val next = input[index + 1]
                    output += next.copy(startMs = segment.startMs)
                    index++
                }
            } else {
                val previous = output.lastOrNull()
                if (previous != null && previous.label == segment.label) {
                    output[output.lastIndex] = previous.copy(
                        endMs = segment.endMs,
                        confidence = weightedConfidence(previous, segment),
                        root = segment.root ?: previous.root,
                        bass = segment.bass ?: previous.bass,
                        pitchClasses = (previous.pitchClasses + segment.pitchClasses).distinct(),
                    )
                } else {
                    output += segment
                }
            }
            index++
        }
        return output.filter { it.endMs > it.startMs }
    }

    private fun weightedConfidence(a: SongChordSegment, b: SongChordSegment): Double {
        val ad = max(1L, a.endMs - a.startMs)
        val bd = max(1L, b.endMs - b.startMs)
        return (a.confidence * ad + b.confidence * bd) / (ad + bd).toDouble()
    }

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

    private fun modeString(values: List<String>): String? = values
        .groupingBy { it }
        .eachCount()
        .maxByOrNull { it.value }
        ?.key

    private val sharpNames = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    private val flatNames = arrayOf("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
    private fun noteName(pc: Int): String = if (preferFlats) flatNames[pc] else sharpNames[pc]

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

private fun List<Double>.averageOrZero(): Double = if (isEmpty()) 0.0 else average()
