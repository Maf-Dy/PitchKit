package com.nicos.pitchkit.tuner.harmony.animechord

import kotlin.math.exp
import kotlin.math.ln

/**
 * The author's HMM smoothing, as a streaming O(T·C) Viterbi.
 *
 * `chord_transcription.hmm` builds `make_sticky_transition(C, stay_prob)` — `stay_prob`
 * on the diagonal, `(1 - stay_prob) / (C - 1)` **uniform** everywhere else — and runs a
 * dense O(T·C²) Viterbi over `log(softmax(logits) + 1e-9)` with a uniform initial
 * distribution. Because the off-diagonal is uniform, the best predecessor of any state is
 * either itself or the single best previous state, which collapses the inner loop:
 * `anime-windowed-report.md` §8.2 verified the reduction path-identical to
 * `hmm.viterbi_jit` on all three heads of a real song (agreement 1.000000).
 *
 * This is the shape `CremaSongViterbiDecoder` and `LvChordiaDictionaryDecoder` already
 * decode in — best/second-best, uniform prior, packed stay bits — instantiated at
 * `classCount = 745`, `logStay = 0` and `logSwitch = ln(1e-9)`, which is exactly what the
 * author's `stay_prob = 1.0` reduces to. Those two are coupled to their own head layouts,
 * so the recurrence is written out here rather than bent around them; it is pinned
 * against a brute-force dense Viterbi and against the reference's own path in
 * `AnimeChordViterbiTest`.
 *
 * Two details are reproduced deliberately, because they decide the path:
 *
 *  * **The emission is quantised to float32 before the epsilon.** The reference softmaxes
 *    in float32, so a probability that underflows becomes exactly 0 and its emission is
 *    the floor `ln(1e-9) = -20.723`. Computed in double it would be −60 or lower, and
 *    with 745 competing classes that changes which state survives.
 *  * **The score accumulator is float32**, as the reference's numba kernel declares it.
 *
 * Frames are pushed in as they are committed, so the whole-song root-chord logits — 128 KB
 * per audio-second — never have to be held; only the 745-bit stay mask per frame does,
 * which is 4 KB per audio-second.
 */
internal class AnimeChordViterbi(
    private val classCount: Int,
    private val logStay: Double = AnimeChordContract.LOG_STAY,
    private val logSwitch: Double = AnimeChordContract.LOG_SWITCH,
    expectedFrames: Int = 1024,
) {
    private var previous = FloatArray(classCount)
    private var current = FloatArray(classCount)
    private val emission = DoubleArray(classCount)
    private val logInitial = ln(1.0 / classCount + AnimeChordContract.EMISSION_EPSILON)

    private var best1 = IntArray(expectedFrames.coerceAtLeast(1))
    private var best2 = IntArray(expectedFrames.coerceAtLeast(1))
    private var stayed = PackedBits(expectedFrames.coerceAtLeast(1).toLong() * classCount)
    private var stayedCapacity = expectedFrames.coerceAtLeast(1)

    var frameCount: Int = 0
        private set

    init {
        require(classCount > 1)
    }

    /** @param logits `classCount` raw head logits for one frame, starting at [offset]. */
    fun add(logits: FloatArray, offset: Int = 0) {
        require(offset >= 0 && offset + classCount <= logits.size)
        logSoftmax(logits, offset)

        if (frameCount == 0) {
            for (index in 0 until classCount) {
                previous[index] = (logInitial + emission[index]).toFloat()
            }
            grow(1)
            best1[0] = 0
            best2[0] = 0
            frameCount = 1
            return
        }

        var bestIndex = 0
        var bestValue = previous[0]
        for (index in 1 until classCount) {
            if (previous[index] > bestValue) {
                bestValue = previous[index]
                bestIndex = index
            }
        }
        var secondIndex = -1
        var secondValue = Float.NEGATIVE_INFINITY
        for (index in 0 until classCount) {
            if (index == bestIndex) continue
            if (secondIndex < 0 || previous[index] > secondValue) {
                secondValue = previous[index]
                secondIndex = index
            }
        }

        val frame = frameCount
        grow(frame + 1)
        best1[frame] = bestIndex
        best2[frame] = secondIndex
        val bitBase = frame.toLong() * classCount

        for (index in 0 until classCount) {
            val other = if (index != bestIndex) bestIndex else secondIndex
            val switchScore = previous[other].toDouble() + logSwitch
            val stayScore = previous[index].toDouble() + logStay
            // The reference breaks an exact tie towards the lower state index.
            val stay = stayScore > switchScore ||
                (stayScore == switchScore && index < other)
            current[index] = ((if (stay) stayScore else switchScore) + emission[index]).toFloat()
            if (stay) stayed.set(bitBase + index)
        }

        val swap = previous
        previous = current
        current = swap
        frameCount = frame + 1
    }

    /** The maximum-likelihood state sequence over every frame added so far. */
    fun decode(): IntArray {
        if (frameCount == 0) return IntArray(0)
        val path = IntArray(frameCount)
        var last = 0
        var bestValue = previous[0]
        for (index in 1 until classCount) {
            if (previous[index] > bestValue) {
                bestValue = previous[index]
                last = index
            }
        }
        path[frameCount - 1] = last
        for (frame in frameCount - 1 downTo 1) {
            val state = path[frame]
            path[frame - 1] = if (stayed.get(frame.toLong() * classCount + state)) {
                state
            } else if (state != best1[frame]) {
                best1[frame]
            } else {
                best2[frame]
            }
        }
        return path
    }

    /**
     * `log(softmax(logits) + 1e-9)` with the softmax rounded to float32 first, which is
     * what the reference computes and what puts an underflowed class on the `ln(1e-9)`
     * floor instead of somewhere far below it.
     */
    private fun logSoftmax(logits: FloatArray, offset: Int) {
        var maximum = logits[offset]
        for (index in 1 until classCount) {
            val value = logits[offset + index]
            if (value > maximum) maximum = value
        }
        var total = 0.0
        for (index in 0 until classCount) {
            val value = exp((logits[offset + index] - maximum).toDouble())
            emission[index] = value
            total += value
        }
        for (index in 0 until classCount) {
            val probability = (emission[index] / total).toFloat()
            emission[index] = ln(probability.toDouble() + AnimeChordContract.EMISSION_EPSILON)
        }
    }

    private fun grow(frames: Int) {
        if (frames <= stayedCapacity) return
        var capacity = stayedCapacity
        while (capacity < frames) capacity *= 2
        best1 = best1.copyOf(capacity)
        best2 = best2.copyOf(capacity)
        val replacement = PackedBits(capacity.toLong() * classCount)
        replacement.copyFrom(stayed)
        stayed = replacement
        stayedCapacity = capacity
    }

    private class PackedBits(bitCount: Long) {
        val words: LongArray

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

        fun copyFrom(other: PackedBits) {
            other.words.copyInto(words)
        }
    }
}
