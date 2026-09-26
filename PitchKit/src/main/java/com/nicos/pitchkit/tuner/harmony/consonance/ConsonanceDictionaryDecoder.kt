package com.nicos.pitchkit.tuner.harmony.consonance

import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaLabelFormatter
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.ln

internal data class ConsonanceDictionaryCandidate(
    val rawLabel: String,
    val displayLabel: String?,
    /** 0..11, or 12 for the no-chord state. */
    val root: Int,
    /** Absolute pitch class 0..11, or 12 for the no-chord state. */
    val bass: Int,
    /** 12-bit mask of absolute pitch classes; bit i set means pitch class i sounds. */
    val mask: Int,
)

internal data class ConsonanceDictionary(
    val transitionPenalty: Double,
    /** Prior weight of "this chord has no member of the group", in [ConsonanceGroups] order. */
    val groupNonePriors: DoubleArray,
    val candidates: List<ConsonanceDictionaryCandidate>,
    /** The likelihood the asset declares; callers may override it per analysis. */
    val likelihood: ConsonanceLikelihood = ConsonanceLikelihood.COMPETITIVE,
) {
    override fun equals(other: Any?): Boolean = this === other ||
        (other is ConsonanceDictionary &&
            transitionPenalty == other.transitionPenalty &&
            groupNonePriors.contentEquals(other.groupNonePriors) &&
            candidates == other.candidates &&
            likelihood == other.likelihood)

    override fun hashCode(): Int {
        var result = transitionPenalty.hashCode()
        result = 31 * result + groupNonePriors.contentHashCode()
        result = 31 * result + candidates.hashCode()
        return 31 * result + likelihood.hashCode()
    }
}

/**
 * Mutually exclusive interval groups relative to the candidate root, mirroring
 * tools/accuracy_audit/consonance_dictionary_decode.py. Within a group the model is asked
 * which member it prefers instead of being rewarded for the absence of the others.
 */
internal object ConsonanceGroups {
    val NAMES = arrayOf("third", "fifth", "seventh")
    val OFFSETS = arrayOf(intArrayOf(3, 4), intArrayOf(6, 7, 8), intArrayOf(10, 11))
    /** Everything the groups do not cover stays Bernoulli: the root plus b9/9, 4/11, 6/13. */
    val REST_OFFSETS = (0 until 12).filter { offset -> OFFSETS.none { offset in it } }.toIntArray()
}

internal data class ConsonanceDictionarySegment(
    val candidate: ConsonanceDictionaryCandidate,
    val startFrame: Int,
    /** Exclusive. */
    val endFrame: Int,
    /** Mean root posterior over the segment's frames. */
    val confidence: Double,
)

internal object ConsonanceDictionaryParser {
    fun parse(json: String, preferFlats: Boolean): ConsonanceDictionary {
        val root = JSONObject(json)
        require(root.getInt("schema_version") == 1)
        // Harte parsing happens offline in tools/accuracy_audit/export_consonance_dictionary.py
        // (mir_eval.chord.encode), so the asset already carries root/bass/pitch-class masks.
        val formatter = LvChordiaLabelFormatter(preferFlats)
        val array = root.getJSONArray("candidates")
        val candidates = List(array.length()) { index ->
            val item = array.getJSONObject(index)
            val raw = item.getString("label")
            ConsonanceDictionaryCandidate(
                rawLabel = raw,
                displayLabel = if (raw == "N" || raw == "X") null else formatter.format(raw),
                root = item.getInt("root"),
                bass = item.getInt("bass"),
                mask = item.getInt("mask"),
            )
        }
        require(candidates.isNotEmpty())
        val likelihood = ConsonanceLikelihood.requireFromId(root.getString("likelihood"))
        val priors = root.getJSONObject("group_none_priors")
        return ConsonanceDictionary(
            transitionPenalty = root.getDouble("transition_penalty"),
            groupNonePriors = DoubleArray(ConsonanceGroups.NAMES.size) {
                priors.getDouble(ConsonanceGroups.NAMES[it])
            },
            candidates = candidates,
            likelihood = likelihood,
        )
    }
}

/**
 * Dictionary Viterbi over the decomposed heads, replicating the likelihood lanes of
 * tools/accuracy_audit/consonance_dictionary_decode.py:
 *
 * - [ConsonanceLikelihood.BERNOULLI]: log p(root) + log p(bass) plus an independent
 *   Bernoulli term over all twelve pitch classes.
 * - [ConsonanceLikelihood.COMPETITIVE]: log p(root) + log p(bass), a Bernoulli term over the
 *   pitch classes outside the third/fifth/seventh groups, and one competitive softmax per
 *   group where the candidate's member activations are weighed against the whole group plus
 *   a prior for having no member at all. The no-chord candidate keeps the all-absent
 *   Bernoulli score.
 * - [ConsonanceLikelihood.COMPETITIVE_BASS]: competitive, with the experimental bass rule -
 *   a root-position candidate keeps at least half the credit of its best other chord tone,
 *   so a passing bass note no longer drags the root away. Inversions and N are unchanged.
 *
 * A uniform chord-change penalty enforces temporal consistency.
 *
 * The forward pass never materializes a frames x candidates likelihood matrix;
 * each frame's row is recomputed on the fly and only the running score vector,
 * the packed per-frame switch bits and the per-frame best previous state survive.
 */
internal class ConsonanceDictionaryDecoder(
    private val dictionary: ConsonanceDictionary,
    private val transitionPenalty: Double = ConsonanceContract.TRANSITION_PENALTY,
    private val likelihood: ConsonanceLikelihood = dictionary.likelihood,
) {
    private val candidates = dictionary.candidates
    private val candidateCount = candidates.size
    private val rootIndex = IntArray(candidateCount) { candidates[it].root }
    private val bassIndex = IntArray(candidateCount) { candidates[it].bass }
    private val maskIndex = IntArray(candidateCount) { candidates[it].mask }
    private val nonePriors = dictionary.groupNonePriors

    /** Bit i set means the candidate contains ConsonanceGroups.REST_OFFSETS[i] above its root. */
    private val restPresence = IntArray(candidateCount)
    /** Per candidate and group, the bits of ConsonanceGroups.OFFSETS[group] the candidate holds. */
    private val groupPresence = IntArray(candidateCount * GROUP_COUNT)
    private val isNoChord = BooleanArray(candidateCount) { candidates[it].rawLabel == "N" }

    /**
     * For [ConsonanceLikelihood.COMPETITIVE_BASS]: the chord tones other than the root of a
     * root-position candidate, or 0 for inversions and the no-chord state, which keep the
     * plain log p(bass) term.
     */
    private val isRootPosition = BooleanArray(candidateCount) { index ->
        !isNoChord[index] && candidates[index].bass == candidates[index].root
    }
    private val rootPositionOthers = IntArray(candidateCount) { index ->
        if (isRootPosition[index]) {
            candidates[index].mask and (1 shl candidates[index].root).inv()
        } else {
            0
        }
    }

    // Scratch reused across frames: the group denominators and the log numerator of every
    // possible member subset, both of which depend only on the root and the frame.
    private val groupLogTotal = DoubleArray(ConsonanceContract.PITCH_COUNT * GROUP_COUNT)
    private val subsetLogNumerator = DoubleArray(ConsonanceContract.PITCH_COUNT * SUBSET_STRIDE)
    private val pitchProbability = DoubleArray(ConsonanceContract.PITCH_COUNT)
    private val onLog = DoubleArray(ConsonanceContract.PITCH_COUNT)
    private val offLog = DoubleArray(ConsonanceContract.PITCH_COUNT)
    private val rootLog = DoubleArray(ConsonanceContract.ROOT_COUNT)
    private val bassLog = DoubleArray(ConsonanceContract.BASS_COUNT)
    private val bassProbability = DoubleArray(ConsonanceContract.BASS_COUNT)

    init {
        require(candidateCount > 0)
        require(rootIndex.all { it in 0 until ConsonanceContract.ROOT_COUNT })
        require(bassIndex.all { it in 0 until ConsonanceContract.BASS_COUNT })
        require(maskIndex.all { it in 0 until (1 shl ConsonanceContract.PITCH_COUNT) })
        require(nonePriors.size == GROUP_COUNT)
        for (candidate in 0 until candidateCount) {
            if (isNoChord[candidate]) continue
            val root = rootIndex[candidate]
            require(root < ConsonanceContract.PITCH_COUNT)
            val mask = maskIndex[candidate]
            var rest = 0
            ConsonanceGroups.REST_OFFSETS.forEachIndexed { index, offset ->
                if (mask and (1 shl ((root + offset) % 12)) != 0) rest = rest or (1 shl index)
            }
            restPresence[candidate] = rest
            for (group in 0 until GROUP_COUNT) {
                var present = 0
                ConsonanceGroups.OFFSETS[group].forEachIndexed { index, offset ->
                    if (mask and (1 shl ((root + offset) % 12)) != 0) present = present or (1 shl index)
                }
                groupPresence[candidate * GROUP_COUNT + group] = present
            }
        }
    }

    /** Per-frame candidate indices for the maximum-likelihood state sequence. */
    fun decodeFrames(heads: ConsonanceHeads): IntArray {
        val frames = heads.frames
        if (frames <= 0) return IntArray(0)
        require(heads.root.size >= frames * ConsonanceContract.ROOT_COUNT)
        require(heads.bass.size >= frames * ConsonanceContract.BASS_COUNT)
        require(heads.pitch.size >= frames * ConsonanceContract.PITCH_COUNT)

        val score = DoubleArray(candidateCount)
        val likelihood = DoubleArray(candidateCount)
        val bestPrevious = IntArray(frames)
        val switched = PackedBits(frames.toLong() * candidateCount.toLong())

        logLikelihoods(heads, frame = 0, output = score)
        for (frame in 1 until frames) {
            var best = 0
            for (candidate in 1 until candidateCount) {
                if (score[candidate] > score[best]) best = candidate
            }
            val viaSwitch = score[best] - transitionPenalty
            bestPrevious[frame] = best
            val switchOffset = frame.toLong() * candidateCount.toLong()

            logLikelihoods(heads, frame, likelihood)
            for (candidate in 0 until candidateCount) {
                // Staying beats switching on equality, matching the reference's strict >.
                if (viaSwitch > score[candidate]) {
                    switched.set(switchOffset + candidate)
                    score[candidate] = viaSwitch
                }
                score[candidate] += likelihood[candidate]
            }
        }

        var state = 0
        for (candidate in 1 until candidateCount) {
            if (score[candidate] > score[state]) state = candidate
        }
        val path = IntArray(frames)
        for (frame in frames - 1 downTo 0) {
            path[frame] = state
            if (frame > 0 && switched.get(frame.toLong() * candidateCount + state)) {
                state = bestPrevious[frame]
            }
        }
        return path
    }

    /** Groups consecutive identical states; no minimum-duration post-processing. */
    fun decode(heads: ConsonanceHeads): List<ConsonanceDictionarySegment> {
        val path = decodeFrames(heads)
        if (path.isEmpty()) return emptyList()
        val segments = mutableListOf<ConsonanceDictionarySegment>()
        var start = 0
        while (start < path.size) {
            var end = start + 1
            while (end < path.size && path[end] == path[start]) end++
            val candidate = candidates[path[start]]
            var sum = 0.0
            for (frame in start until end) {
                sum += softmaxProbability(
                    heads.root,
                    frame * ConsonanceContract.ROOT_COUNT,
                    ConsonanceContract.ROOT_COUNT,
                    candidate.root,
                )
            }
            segments += ConsonanceDictionarySegment(
                candidate = candidate,
                startFrame = start,
                endFrame = end,
                confidence = sum / (end - start).toDouble(),
            )
            start = end
        }
        return segments
    }

    private fun logLikelihoods(heads: ConsonanceHeads, frame: Int, output: DoubleArray) {
        logSoftmaxInto(
            heads.root,
            frame * ConsonanceContract.ROOT_COUNT,
            ConsonanceContract.ROOT_COUNT,
            probabilities = null,
            logarithms = rootLog,
        )
        logSoftmaxInto(
            heads.bass,
            frame * ConsonanceContract.BASS_COUNT,
            ConsonanceContract.BASS_COUNT,
            probabilities = bassProbability,
            logarithms = bassLog,
        )
        val pitchOffset = frame * ConsonanceContract.PITCH_COUNT
        var allOff = 0.0
        for (pc in 0 until ConsonanceContract.PITCH_COUNT) {
            val probability = sigmoid(heads.pitch[pitchOffset + pc].toDouble())
            pitchProbability[pc] = probability
            onLog[pc] = ln(probability + EPSILON)
            offLog[pc] = ln(1.0 - probability + EPSILON)
            allOff += offLog[pc]
        }
        if (likelihood != ConsonanceLikelihood.BERNOULLI) prepareGroups()

        for (candidate in 0 until candidateCount) {
            val root = rootIndex[candidate]
            var value: Double
            if (likelihood == ConsonanceLikelihood.BERNOULLI) {
                // The reference sums the present and absent pitch classes as two separate
                // matrix products, so the two halves are accumulated separately here too.
                val mask = maskIndex[candidate]
                var on = 0.0
                var off = 0.0
                for (pc in 0 until ConsonanceContract.PITCH_COUNT) {
                    if (mask and (1 shl pc) != 0) on += onLog[pc] else off += offLog[pc]
                }
                value = on + off
            } else if (isNoChord[candidate]) {
                // The no-chord option keeps its all-absent Bernoulli scoring.
                value = allOff
            } else {
                value = 0.0
                val rest = restPresence[candidate]
                ConsonanceGroups.REST_OFFSETS.forEachIndexed { index, offset ->
                    val pc = (root + offset) % 12
                    value += if (rest and (1 shl index) != 0) onLog[pc] else offLog[pc]
                }
                for (group in 0 until GROUP_COUNT) {
                    val present = groupPresence[candidate * GROUP_COUNT + group]
                    value += subsetLogNumerator[root * SUBSET_STRIDE + SUBSET_BASE[group] + present] -
                        groupLogTotal[root * GROUP_COUNT + group]
                }
            }
            value += rootLog[rootIndex[candidate]]
            value += bassTerm(candidate)
            // The reference stores the likelihood row as float32 before the Viterbi
            // accumulation; rounding here keeps the accumulated scores identical.
            output[candidate] = value.toFloat().toDouble()
        }
    }

    /**
     * log p(bass) for the candidate, with the experimental root-position allowance of the
     * `competitive-bass` lane: a root-position chord keeps at least half the bass posterior
     * of its best other chord tone, so the root is not dragged onto a passing bass note.
     */
    private fun bassTerm(candidate: Int): Double {
        if (likelihood != ConsonanceLikelihood.COMPETITIVE_BASS || !isRootPosition[candidate]) {
            return bassLog[bassIndex[candidate]]
        }
        var alternative = 0.0
        var others = rootPositionOthers[candidate]
        while (others != 0) {
            val pc = Integer.numberOfTrailingZeros(others)
            alternative = maxOf(alternative, bassProbability[pc])
            others = others and (others - 1)
        }
        // An empty "others" set leaves the alternative at zero, matching the reference.
        val root = rootIndex[candidate]
        return ln(maxOf(bassProbability[root], 0.5 * alternative) + EPSILON)
    }

    /**
     * Per root and group: the log denominator (all member activations plus the no-member
     * prior) and the log numerator of every possible member subset, including the empty
     * subset which scores the prior itself.
     */
    private fun prepareGroups() {
        for (root in 0 until ConsonanceContract.PITCH_COUNT) {
            for (group in 0 until GROUP_COUNT) {
                val offsets = ConsonanceGroups.OFFSETS[group]
                var total = 0.0
                for (offset in offsets) total += pitchProbability[(root + offset) % 12]
                groupLogTotal[root * GROUP_COUNT + group] = ln(total + nonePriors[group] + EPSILON)
                val base = root * SUBSET_STRIDE + SUBSET_BASE[group]
                for (subset in 0 until (1 shl offsets.size)) {
                    var numerator = if (subset == 0) nonePriors[group] else 0.0
                    var bits = subset
                    while (bits != 0) {
                        val index = Integer.numberOfTrailingZeros(bits)
                        numerator += pitchProbability[(root + offsets[index]) % 12]
                        bits = bits and (bits - 1)
                    }
                    subsetLogNumerator[base + subset] = ln(numerator + EPSILON)
                }
            }
        }
    }

    private fun logSoftmaxInto(
        values: FloatArray,
        offset: Int,
        width: Int,
        probabilities: DoubleArray?,
        logarithms: DoubleArray,
    ) {
        var maximum = Double.NEGATIVE_INFINITY
        for (index in 0 until width) maximum = maxOf(maximum, values[offset + index].toDouble())
        var denominator = 0.0
        for (index in 0 until width) denominator += exp(values[offset + index].toDouble() - maximum)
        for (index in 0 until width) {
            val probability = exp(values[offset + index].toDouble() - maximum) / denominator
            probabilities?.set(index, probability)
            logarithms[index] = ln(probability + EPSILON)
        }
    }

    private fun softmaxProbability(
        values: FloatArray,
        offset: Int,
        width: Int,
        selected: Int,
    ): Double {
        if (selected !in 0 until width) return 0.0
        var maximum = Double.NEGATIVE_INFINITY
        for (index in 0 until width) maximum = maxOf(maximum, values[offset + index].toDouble())
        var denominator = 0.0
        for (index in 0 until width) denominator += exp(values[offset + index].toDouble() - maximum)
        return if (denominator > 0.0) {
            exp(values[offset + selected].toDouble() - maximum) / denominator
        } else {
            0.0
        }
    }

    private fun sigmoid(value: Double): Double = 1.0 / (1.0 + exp(-value))

    private companion object {
        const val EPSILON = 1e-9
        const val GROUP_COUNT = 3
        /** Room for one log numerator per member subset of every group, per root. */
        const val SUBSET_STRIDE = 16
        val SUBSET_BASE = intArrayOf(0, 4, 12)
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
            words[word] = words[word] or (1L shl (index and 63L).toInt())
        }

        fun get(index: Long): Boolean {
            val word = (index ushr 6).toInt()
            return words[word] and (1L shl (index and 63L).toInt()) != 0L
        }
    }
}
