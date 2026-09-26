package com.nicos.pitchkit.tuner.harmony.chordformer

import kotlin.math.max
import kotlin.math.min

/**
 * How the whole-song feature matrix is cut into the fixed 2048-frame windows the exported
 * graph accepts.
 *
 * This reproduces the PC reference lane (`fold3-w2048o512-submission`) exactly: window 0
 * starts at real frame 0, reads 2048 real frames and commits `[0, 2048 - half)`; every
 * later window starts a step of `WINDOW_FRAMES - OVERLAP_FRAMES` further on and commits
 * only its centre, so each of those frames carries at least
 * [ChordFormerContract.HALF_OVERLAP_FRAMES] frames of left context and, unless it is at
 * the very end of the song, as much right context.
 *
 * The port used to prime the first window: start it at `-half` and reflect the opening of
 * the song into the lead-in. `.accuracy-work/annotations/chordformer-phone-diagnosis.md`
 * measured that at 2048/512 and it *causes* the failure it was meant to prevent (-5.2 root
 * / -4.8 family pooled). The mechanism is arithmetic, not acoustic: a primed window 0
 * covers real frames `[0, 1792)` - 41.6 s - instead of `[0, 2048)` - 47.6 s, and on a track
 * with a sparse intro those last 5.9 s are the only music the window has to read, so the
 * model answers `N` across the intro exactly as a shorter window did. What fills the
 * lead-in (reflection, the dB floor, an edge repeat) scored identically; only the lost real
 * right-context mattered.
 *
 * Pure arithmetic, kept apart from the ONNX session so it can be tested on the JVM.
 */
internal object ChordFormerWindowPlan {
    /**
     * @param startFrame first feature frame the window covers; never negative.
     * @param usableFrames how many of the window's [ChordFormerContract.WINDOW_FRAMES]
     * slots carry feature data; the rest are padded with the dB floor.
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
        val committedFrames: IntRange get() = (startFrame + commitStart) until (startFrame + commitEnd)
    }

    fun windows(frameCount: Int): List<Window> {
        require(frameCount >= 0)
        if (frameCount == 0) return emptyList()

        val window = ChordFormerContract.WINDOW_FRAMES
        val half = ChordFormerContract.HALF_OVERLAP_FRAMES
        val step = window - ChordFormerContract.OVERLAP_FRAMES

        val result = mutableListOf<Window>()
        var startFrame = 0
        while (startFrame < frameCount) {
            val endFrame = min(frameCount, startFrame + window)
            val usableFrames = endFrame - startFrame
            val isLast = endFrame == frameCount
            // The song's very first frame has no earlier window that could have committed
            // it, so window 0 keeps its own frame 0 rather than a primed lead-in.
            val commitStart = if (startFrame == 0) 0 else half
            val commitEnd = if (isLast) usableFrames else max(commitStart, usableFrames - half)
            result += Window(startFrame, usableFrames, commitStart, commitEnd, isLast)
            if (isLast) break
            startFrame += step
        }
        return result
    }

    /**
     * Where window slot [frame] reads from. Every slot a window declares usable is a real
     * frame of the song; the clamp is only a guard, since the plan never runs a slot past
     * the end of the feature matrix.
     */
    fun sourceFrame(frame: Int, frameCount: Int): Int {
        require(frameCount > 0)
        require(frame >= 0) { "The window plan is unprimed; slot $frame is not a real frame" }
        return min(frame, frameCount - 1)
    }
}
