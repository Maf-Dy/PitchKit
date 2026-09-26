package com.nicos.pitchkit.tuner.harmony.animechord

import kotlin.math.max
import kotlin.math.min

/**
 * How the whole-song feature matrix is cut into the 30 s windows the graph is run at.
 *
 * Frame for frame the plan `ChordFormerWindowPlan` uses, and frame for frame the plan the
 * PC lane `anime-hmm-w30-o50` measured: `step = window - overlap`, `half = overlap / 2`,
 * window 0 starts at real frame 0 and commits from **its own frame 0 — unprimed**, every
 * later window commits only `[half, usable - half)`, and the last window commits to its
 * end. Every frame of every song is committed exactly once, by the window that saw the
 * most context around it.
 *
 * Unlike ChordFormer's graph, this one has a dynamic time axis, so a short final window is
 * run at its real length instead of being padded — there is no pad value to argue about.
 *
 * The first window is deliberately not primed. `chordformer-phone-diagnosis.md` measured
 * priming at -5.2 root / -4.8 family for the same construction, because a primed window 0
 * buys synthetic left context by giving up real right context in the one window that has
 * to read an intro, and `anime-windowed-report.md` §4.1 confirms the unprimed opening on
 * this model.
 *
 * Pure arithmetic, kept apart from the ONNX session so it can be tested on the JVM.
 */
internal object AnimeChordWindowPlan {
    /**
     * @param startFrame first feature frame the window covers; never negative.
     * @param usableFrames how many real frames the window reads. The graph's time axis is
     * dynamic, so this is also the tensor length.
     * @param commitStart first window-local frame whose heads are kept; 0 for the first
     * window, which has no earlier window to defer to, and a half overlap for the rest.
     * @param commitEnd one past the last window-local frame whose heads are kept.
     */
    data class Window(
        val startFrame: Int,
        val usableFrames: Int,
        val commitStart: Int,
        val commitEnd: Int,
        val isLast: Boolean,
    ) {
        val committedFrames: IntRange
            get() = (startFrame + commitStart) until (startFrame + commitEnd)
    }

    fun windows(frameCount: Int): List<Window> = windows(
        frameCount = frameCount,
        window = AnimeChordContract.WINDOW_FRAMES,
        overlap = AnimeChordContract.OVERLAP_FRAMES,
    )

    fun windows(frameCount: Int, window: Int, overlap: Int): List<Window> {
        require(frameCount >= 0)
        require(window > 0)
        require(overlap in 0 until window)
        if (frameCount == 0) return emptyList()

        val half = overlap / 2
        val step = window - overlap

        val result = mutableListOf<Window>()
        var startFrame = 0
        while (startFrame < frameCount) {
            val endFrame = min(frameCount, startFrame + window)
            val usableFrames = endFrame - startFrame
            val isLast = endFrame == frameCount
            val commitStart = if (startFrame == 0) 0 else half
            val commitEnd = if (isLast) usableFrames else max(commitStart, usableFrames - half)
            result += Window(startFrame, usableFrames, commitStart, commitEnd, isLast)
            if (isLast) break
            startFrame += step
        }
        return result
    }
}
