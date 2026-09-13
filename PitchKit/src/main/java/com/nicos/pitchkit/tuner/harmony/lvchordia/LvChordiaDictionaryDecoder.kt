package com.nicos.pitchkit.tuner.harmony.lvchordia

import org.json.JSONObject
import kotlin.math.ln

internal data class LvChordiaDictionaryCandidate(
    val rawLabel: String,
    val displayLabel: String?,
    val triad: Int,
    val bass: Int,
    val seventh: Int,
    val ninth: Int,
    val eleventh: Int,
    val thirteenth: Int,
)

internal data class LvChordiaDictionary(
    val transitionPenalty: Double,
    val candidates: List<LvChordiaDictionaryCandidate>,
)

internal data class LvChordiaDecodedFrame(
    val label: String?,
    val confidence: Double,
)

internal object LvChordiaDictionaryParser {
    fun parse(json: String, preferFlats: Boolean): LvChordiaDictionary {
        val root = JSONObject(json)
        require(root.getInt("schema_version") == 1)
        val formatter = LvChordiaLabelFormatter(preferFlats)
        val array = root.getJSONArray("candidates")
        val candidates = List(array.length()) { index ->
            val item = array.getJSONObject(index)
            val raw = item.getString("label")
            LvChordiaDictionaryCandidate(
                rawLabel = raw,
                displayLabel = if (raw == "N" || raw == "X") null else formatter.format(raw),
                triad = item.getInt("triad"),
                bass = item.getInt("bass"),
                seventh = item.getInt("seventh"),
                ninth = item.getInt("ninth"),
                eleventh = item.getInt("eleventh"),
                thirteenth = item.getInt("thirteenth"),
            )
        }
        require(candidates.isNotEmpty())
        require(candidates.first().rawLabel == "N") {
            "LV-Chordia full dictionary must start with N, matching upstream HMM initialization"
        }
        return LvChordiaDictionary(
            transitionPenalty = root.getDouble("transition_penalty"),
            candidates = candidates,
        )
    }
}

/** Memory-efficient port of LV-Chordia's full-dictionary no-beat XHMM decoder. */
internal class LvChordiaDictionaryDecoder(
    private val dictionary: LvChordiaDictionary,
) {
    private val candidates = dictionary.candidates
    private val candidateCount = candidates.size
    private val triadIndex = IntArray(candidateCount) { candidates[it].triad }
    private val bassIndex = IntArray(candidateCount) { candidates[it].bass + 1 }
    private val seventhIndex = IntArray(candidateCount) { candidates[it].seventh }
    private val ninthIndex = IntArray(candidateCount) { candidates[it].ninth }
    private val eleventhIndex = IntArray(candidateCount) { candidates[it].eleventh }
    private val thirteenthIndex = IntArray(candidateCount) { candidates[it].thirteenth }

    init {
        require(candidateCount > 0)
        require(triadIndex.all { it in 0 until LvChordiaContract.TRIAD_COUNT })
        require(bassIndex.all { it in 0 until LvChordiaContract.BASS_COUNT })
    }

    fun decode(heads: LvChordiaHeads): List<LvChordiaDecodedFrame> {
        val frames = heads.frames
        if (frames <= 0) return emptyList()
        val logs = LoggedHeads(heads)

        var previous = DoubleArray(candidateCount) { Double.NEGATIVE_INFINITY }
        var current = DoubleArray(candidateCount)
        val globalBest = IntArray(frames)
        val stayed = PackedBits(frames.toLong() * candidateCount.toLong())

        previous[0] = observationLog(logs, frame = 0, candidateIndex = 0)
        globalBest[0] = 0

        for (frame in 1 until frames) {
            val previousBestIndex = globalBest[frame - 1]
            val jumpScore = previous[previousBestIndex] - dictionary.transitionPenalty
            var bestCurrentIndex = 0
            var bestCurrentValue = Double.NEGATIVE_INFINITY

            val triadOffset = frame * LvChordiaContract.TRIAD_COUNT
            val bassOffset = frame * LvChordiaContract.BASS_COUNT
            val seventhOffset = frame * LvChordiaContract.SEVENTH_COUNT
            val ninthOffset = frame * LvChordiaContract.NINTH_COUNT
            val eleventhOffset = frame * LvChordiaContract.ELEVENTH_COUNT
            val thirteenthOffset = frame * LvChordiaContract.THIRTEENTH_COUNT
            val backpointerOffset = frame.toLong() * candidateCount.toLong()

            for (candidate in 0 until candidateCount) {
                var observation = logs.triad[triadOffset + triadIndex[candidate]].toDouble()
                observation += logs.bass[bassOffset + bassIndex[candidate]]

                val s7 = seventhIndex[candidate]
                if (s7 >= 0) observation += logs.seventh[seventhOffset + s7]
                val s9 = ninthIndex[candidate]
                if (s9 >= 0) observation += logs.ninth[ninthOffset + s9]
                val s11 = eleventhIndex[candidate]
                if (s11 >= 0) observation += logs.eleventh[eleventhOffset + s11]
                val s13 = thirteenthIndex[candidate]
                if (s13 >= 0) observation += logs.thirteenth[thirteenthOffset + s13]

                val stayScore = previous[candidate]
                val useStay = stayScore > jumpScore
                val transition = if (useStay) stayScore else jumpScore
                val value = transition + observation
                current[candidate] = value
                if (useStay) stayed.set(backpointerOffset + candidate)
                if (value > bestCurrentValue) {
                    bestCurrentValue = value
                    bestCurrentIndex = candidate
                }
            }

            globalBest[frame] = bestCurrentIndex
            val swap = previous
            previous = current
            current = swap
        }

        val decoded = IntArray(frames)
        decoded[frames - 1] = globalBest[frames - 1]
        for (frame in frames - 1 downTo 1) {
            val candidate = decoded[frame]
            val didStay = stayed.get(frame.toLong() * candidateCount + candidate)
            decoded[frame - 1] = if (didStay) candidate else globalBest[frame - 1]
        }

        return List(frames) { frame ->
            val candidate = candidates[decoded[frame]]
            LvChordiaDecodedFrame(
                label = candidate.displayLabel,
                confidence = observationConfidence(heads, frame, candidate),
            )
        }
    }

    private fun observationLog(logs: LoggedHeads, frame: Int, candidateIndex: Int): Double {
        var score = logs.triad[
            frame * LvChordiaContract.TRIAD_COUNT + triadIndex[candidateIndex]
        ].toDouble()
        score += logs.bass[
            frame * LvChordiaContract.BASS_COUNT + bassIndex[candidateIndex]
        ]
        val s7 = seventhIndex[candidateIndex]
        if (s7 >= 0) score += logs.seventh[frame * LvChordiaContract.SEVENTH_COUNT + s7]
        val s9 = ninthIndex[candidateIndex]
        if (s9 >= 0) score += logs.ninth[frame * LvChordiaContract.NINTH_COUNT + s9]
        val s11 = eleventhIndex[candidateIndex]
        if (s11 >= 0) score += logs.eleventh[frame * LvChordiaContract.ELEVENTH_COUNT + s11]
        val s13 = thirteenthIndex[candidateIndex]
        if (s13 >= 0) score += logs.thirteenth[frame * LvChordiaContract.THIRTEENTH_COUNT + s13]
        return score
    }

    private fun observationConfidence(
        heads: LvChordiaHeads,
        frame: Int,
        candidate: LvChordiaDictionaryCandidate,
    ): Double {
        var logSum = ln(
            probability(heads.triad, frame, LvChordiaContract.TRIAD_COUNT, candidate.triad)
                .coerceAtLeast(1e-12)
        )
        var count = 1

        logSum += ln(
            probability(heads.bass, frame, LvChordiaContract.BASS_COUNT, candidate.bass + 1)
                .coerceAtLeast(1e-12)
        )
        count++

        if (candidate.seventh >= 0) {
            logSum += ln(probability(heads.seventh, frame, LvChordiaContract.SEVENTH_COUNT, candidate.seventh).coerceAtLeast(1e-12))
            count++
        }
        if (candidate.ninth >= 0) {
            logSum += ln(probability(heads.ninth, frame, LvChordiaContract.NINTH_COUNT, candidate.ninth).coerceAtLeast(1e-12))
            count++
        }
        if (candidate.eleventh >= 0) {
            logSum += ln(probability(heads.eleventh, frame, LvChordiaContract.ELEVENTH_COUNT, candidate.eleventh).coerceAtLeast(1e-12))
            count++
        }
        if (candidate.thirteenth >= 0) {
            logSum += ln(probability(heads.thirteenth, frame, LvChordiaContract.THIRTEENTH_COUNT, candidate.thirteenth).coerceAtLeast(1e-12))
            count++
        }
        return kotlin.math.exp(logSum / count.toDouble()).coerceIn(0.0, 1.0)
    }

    private fun probability(
        values: FloatArray,
        frame: Int,
        width: Int,
        index: Int,
    ): Double {
        if (index !in 0 until width) return 0.0
        return values[frame * width + index].toDouble().coerceIn(0.0, 1.0)
    }

    private class LoggedHeads(heads: LvChordiaHeads) {
        val triad = logArray(heads.triad)
        val bass = logArray(heads.bass)
        val seventh = logArray(heads.seventh)
        val ninth = logArray(heads.ninth)
        val eleventh = logArray(heads.eleventh)
        val thirteenth = logArray(heads.thirteenth)

        private companion object {
            fun logArray(values: FloatArray): FloatArray = FloatArray(values.size) { index ->
                ln(values[index].toDouble().coerceAtLeast(1e-12)).toFloat()
            }
        }
    }

    private class PackedBits(bitCount: Long) {
        private val words: LongArray

        init {
            require(bitCount >= 0L)
            val wordCount = ((bitCount + 63L) ushr 6)
                .coerceAtMost(Int.MAX_VALUE.toLong())
                .toInt()
            words = LongArray(wordCount)
        }

        fun set(index: Long) {
            val word = (index ushr 6).toInt()
            val bit = (index and 63L).toInt()
            words[word] = words[word] or (1L shl bit)
        }

        fun get(index: Long): Boolean {
            val word = (index ushr 6).toInt()
            val bit = (index and 63L).toInt()
            return words[word] and (1L shl bit) != 0L
        }
    }
}
