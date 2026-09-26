package com.nicos.pitchkit.tuner.harmony.chordformer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The windowing is the only part of this engine that is not upstream's, so its arithmetic
 * is pinned here: every feature frame is committed exactly once, in order, from a window
 * that read it as a real frame, and window 0 is *not* primed -- it starts at frame 0 and
 * reads 2048 real frames, which is what the PC reference lane does and what
 * `.accuracy-work/annotations/chordformer-phone-diagnosis.md` measured at +5.2 root over
 * the reflect-primed plan this replaced.
 */
class ChordFormerWindowPlanTest {
    private val window = ChordFormerContract.WINDOW_FRAMES
    private val half = ChordFormerContract.HALF_OVERLAP_FRAMES
    private val step = window - ChordFormerContract.OVERLAP_FRAMES

    private val lengths = listOf(
        // shorter than one window ...
        1, 2, half, half + 1, 1000, window - 1,
        // ... exactly one window ...
        window,
        // ... and N windows plus a partial tail.
        window + 1, window + half, window + step, 2 * window, 3 * window + 17,
        window + step + 1, window + 2 * step + 5, 12_345, 40_000,
    )

    @Test
    fun everyFrameIsCommittedExactlyOnceAndInOrder() {
        for (frameCount in lengths) {
            val committed = mutableListOf<Int>()
            for (plan in ChordFormerWindowPlan.windows(frameCount)) {
                assertTrue(
                    "usable frames $frameCount/${plan.usableFrames}",
                    plan.usableFrames in 1..window,
                )
                // A window never reads past the end of the song, so no slot it declares
                // usable needs clamping or priming.
                assertTrue(
                    "frameCount=$frameCount start=${plan.startFrame}",
                    plan.startFrame >= 0 && plan.startFrame + plan.usableFrames <= frameCount,
                )
                committed += plan.committedFrames.toList()
            }
            // No gaps, no overlap, no reordering: exactly 0 until frameCount.
            assertEquals("frameCount=$frameCount", (0 until frameCount).toList(), committed)
        }
    }

    @Test
    fun onlyTheFirstWindowCommitsWithoutAHalfOverlapOfLeftContext() {
        for (frameCount in lengths) {
            val plans = ChordFormerWindowPlan.windows(frameCount)
            for ((index, plan) in plans.withIndex()) {
                assertEquals(
                    "frameCount=$frameCount start=${plan.startFrame}",
                    if (index == 0) 0 else half,
                    plan.commitStart,
                )
                // Only the final window is allowed to commit right up to its last frame,
                // because there is no more song to protect it with.
                if (!plan.isLast) {
                    assertEquals(plan.usableFrames - half, plan.commitEnd)
                    assertEquals(window, plan.usableFrames)
                }
            }
        }
    }

    @Test
    fun theFirstWindowIsNotPrimed() {
        val frameCount = 4 * window
        val first = ChordFormerWindowPlan.windows(frameCount).first()
        assertEquals(0, first.startFrame)
        assertEquals(0, first.commitStart)
        // 2048 real frames read, the full window minus one half overlap committed.
        assertEquals(window, first.usableFrames)
        assertEquals(window - half, first.commitEnd)
        assertEquals(0 until (window - half), first.committedFrames)

        assertEquals(0, ChordFormerWindowPlan.sourceFrame(0, frameCount))
        assertEquals(7, ChordFormerWindowPlan.sourceFrame(7, frameCount))
    }

    @Test
    fun aSongShorterThanOneWindowIsASingleWindowCoveringAllOfIt() {
        for (frameCount in listOf(1, 2, half, 1000, window - 1, window)) {
            val plans = ChordFormerWindowPlan.windows(frameCount)
            assertEquals("frameCount=$frameCount", 1, plans.size)
            val single = plans.single()
            assertTrue(single.isLast)
            assertEquals(0, single.startFrame)
            assertEquals(frameCount, single.usableFrames)
            assertEquals(0, single.commitStart)
            assertEquals(frameCount, single.commitEnd)
            assertEquals((0 until frameCount).toList(), single.committedFrames.toList())
        }
    }

    @Test
    fun aPartialTailIsCoveredByOneExtraWindow() {
        // One full window plus a 17-frame tail: two windows, the second one short.
        val frameCount = window + 17
        val plans = ChordFormerWindowPlan.windows(frameCount)
        assertEquals(2, plans.size)
        assertEquals(0, plans[0].startFrame)
        assertEquals(0 until (window - half), plans[0].committedFrames)
        assertEquals(step, plans[1].startFrame)
        assertTrue(plans[1].isLast)
        assertEquals(frameCount - step, plans[1].usableFrames)
        assertEquals((window - half) until frameCount, plans[1].committedFrames)
    }

    @Test
    fun windowsAdvanceByTheDeclaredStep() {
        val plans = ChordFormerWindowPlan.windows(10 * window)
        for (index in 1 until plans.size) {
            assertEquals(step, plans[index].startFrame - plans[index - 1].startFrame)
        }
        assertEquals(0, plans.first().startFrame)
        assertTrue(plans.last().isLast)
        assertEquals(1, plans.count { it.isLast })
    }

    @Test
    fun anEmptyFeatureMatrixProducesNoWindows() {
        assertTrue(ChordFormerWindowPlan.windows(0).isEmpty())
    }
}
