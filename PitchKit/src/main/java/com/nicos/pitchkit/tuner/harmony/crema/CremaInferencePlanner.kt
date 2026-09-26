package com.nicos.pitchkit.tuner.harmony.crema

import kotlin.math.min

/** One model invocation: the feature frames fed in, and the frames kept from it. */
internal data class CremaInferenceChunk(
    val startFrame: Int,
    val endFrame: Int,
    val commitStart: Int,
    val commitEnd: Int,
) {
    val frameCount: Int get() = endFrame - startFrame
}

/**
 * Chooses how many frames Crema sees per inference call.
 *
 * Crema 0.2.0 is a bidirectional GRU, so every frame's chord tag depends on the
 * whole sequence in both directions, and the exported graph leaves the time axis
 * dynamic (`cqt_mag: [batch, time, 216, 2]`). The accuracy audit's context
 * ablation (tools/accuracy_audit/FINDINGS.md, "Isolated context experiments")
 * replayed identical saved features through the same ONNX graph and measured
 * only 0.830 chord-tag argmax agreement between the 128-frame / 16-overlap
 * windowing this port used to run and a single full-song pass - i.e. the
 * windowing alone changed 17% of the tag frames. Over the same annotations PC
 * Crema scores root 0.794 / family 0.714 where the windowed port scored
 * 0.725 / 0.643. Songs are therefore inferred in a single pass; the windowed
 * path only survives as a safety valve for pathologically long inputs and for
 * runtimes that run out of memory on the single pass.
 */
internal object CremaInferencePlanner {
    /**
     * ~18.6 min at hop 4096 / 44.1 kHz. The HCQT alone is 216 * 2 * 4 bytes per
     * frame (1.7 kB), so this cap holds the feature tensor near 21 MB.
     */
    const val MAX_SINGLE_PASS_FRAMES = 12_000

    /** Fallback context, still ~3.2 min of bidirectional context per call. */
    const val FALLBACK_CHUNK_FRAMES = 2_048

    /** Discarded half on each side of a fallback join, to hide GRU edge effects. */
    const val FALLBACK_OVERLAP_FRAMES = 256

    fun plan(
        frameCount: Int,
        maxSinglePassFrames: Int = MAX_SINGLE_PASS_FRAMES,
    ): List<CremaInferenceChunk> {
        if (frameCount <= 0) return emptyList()
        if (frameCount <= maxSinglePassFrames) {
            return listOf(CremaInferenceChunk(0, frameCount, 0, frameCount))
        }
        return chunked(frameCount)
    }

    /** The windowed path, used when the single pass is refused or fails. */
    fun chunked(
        frameCount: Int,
        chunkFrames: Int = FALLBACK_CHUNK_FRAMES,
        overlapFrames: Int = FALLBACK_OVERLAP_FRAMES,
    ): List<CremaInferenceChunk> {
        if (frameCount <= 0) return emptyList()
        if (frameCount <= chunkFrames) {
            return listOf(CremaInferenceChunk(0, frameCount, 0, frameCount))
        }
        val half = overlapFrames / 2
        val step = chunkFrames - overlapFrames
        require(step > 0) { "Crema fallback overlap must be smaller than the chunk" }

        val chunks = mutableListOf<CremaInferenceChunk>()
        var start = 0
        while (start < frameCount) {
            val end = min(start + chunkFrames, frameCount)
            val last = end >= frameCount
            val commitStart = if (start == 0) start else start + half
            val commitEnd = if (last) end else end - half
            if (commitEnd > commitStart) {
                chunks += CremaInferenceChunk(start, end, commitStart, commitEnd)
            }
            if (last) break
            start += step
        }
        return chunks
    }
}
