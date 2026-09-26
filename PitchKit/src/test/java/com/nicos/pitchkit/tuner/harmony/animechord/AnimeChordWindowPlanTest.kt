package com.nicos.pitchkit.tuner.harmony.animechord

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The windowing is the only part of this engine that is not upstream's, so its arithmetic
 * is pinned here: every feature frame is committed exactly once, in order, by a window
 * that read it as a real frame, and window 0 is *not* primed. That is the plan the PC lane
 * `anime-hmm-w30-o50` measured at 87.0 root / 82.6 family / 68.8 boundary F1 against the
 * whole-song lane's 80.2 / 76.7 / 58.5.
 */
class AnimeChordWindowPlanTest {
    private val window = AnimeChordContract.WINDOW_FRAMES
    private val half = AnimeChordContract.HALF_OVERLAP_FRAMES
    private val step = window - AnimeChordContract.OVERLAP_FRAMES

    private val lengths = listOf(
        1, 2, half, half + 1, 1000, window - 1,
        window,
        window + 1, window + half, window + step, 2 * window, 3 * window + 17,
        window + step + 1, window + 2 * step + 5, 12_345, 31_434,
    )

    @Test
    fun theShippedGeometryIsThirtySecondsAtFiftyPercent() {
        assertEquals(1292, window)
        assertEquals(646, AnimeChordContract.OVERLAP_FRAMES)
        assertEquals(323, half)
        assertEquals(646, step)
        // 30 s at 43.066 frames/s, to the frame.
        assertEquals(
            AnimeChordContract.WINDOW_SECONDS,
            window * AnimeChordContract.SECONDS_PER_FRAME,
            0.02,
        )
    }

    @Test
    fun everyFrameIsCommittedExactlyOnceAndInOrder() {
        for (frameCount in lengths) {
            val committed = mutableListOf<Int>()
            for (plan in AnimeChordWindowPlan.windows(frameCount)) {
                assertTrue(
                    "usable frames $frameCount/${plan.usableFrames}",
                    plan.usableFrames in 1..window,
                )
                assertTrue(
                    "frameCount=$frameCount start=${plan.startFrame}",
                    plan.startFrame >= 0 && plan.startFrame + plan.usableFrames <= frameCount,
                )
                committed += plan.committedFrames.toList()
            }
            assertEquals("frameCount=$frameCount", (0 until frameCount).toList(), committed)
        }
    }

    @Test
    fun onlyTheFirstWindowCommitsWithoutAHalfOverlapOfLeftContext() {
        for (frameCount in lengths) {
            val plans = AnimeChordWindowPlan.windows(frameCount)
            for ((index, plan) in plans.withIndex()) {
                assertEquals(
                    "frameCount=$frameCount start=${plan.startFrame}",
                    if (index == 0) 0 else half,
                    plan.commitStart,
                )
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
        val first = AnimeChordWindowPlan.windows(frameCount).first()
        assertEquals(0, first.startFrame)
        assertEquals(0, first.commitStart)
        assertEquals(window, first.usableFrames)
        assertEquals(window - half, first.commitEnd)
        assertEquals(0 until (window - half), first.committedFrames)
    }

    @Test
    fun aSongShorterThanOneWindowIsASingleWindowCoveringAllOfIt() {
        for (frameCount in listOf(1, 2, half, 1000, window - 1, window)) {
            val plans = AnimeChordWindowPlan.windows(frameCount)
            assertEquals("frameCount=$frameCount", 1, plans.size)
            val single = plans.single()
            assertTrue(single.isLast)
            assertEquals(0, single.startFrame)
            assertEquals(frameCount, single.usableFrames)
            assertEquals(0, single.commitStart)
            assertEquals(frameCount, single.commitEnd)
        }
    }

    @Test
    fun aPartialTailIsCoveredByOneExtraWindow() {
        val frameCount = window + 17
        val plans = AnimeChordWindowPlan.windows(frameCount)
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
        val plans = AnimeChordWindowPlan.windows(10 * window)
        for (index in 1 until plans.size) {
            assertEquals(step, plans[index].startFrame - plans[index - 1].startFrame)
        }
        assertEquals(0, plans.first().startFrame)
        assertTrue(plans.last().isLast)
        assertEquals(1, plans.count { it.isLast })
    }

    @Test
    fun theWindowCountMatchesTheReferenceLane() {
        // `anime-windowed-01/<song>/anime-hmm-w30-o50/result.json` reports these window
        // counts for the twelve annotated songs, from their feature frame counts.
        val reference = mapOf(
            14_434 to 22, 31_434 to 48, 5_284 to 8, 11_103 to 17, 6_617 to 10,
            9_328 to 14, 3_258 to 5, 10_073 to 15, 7_075 to 10, 5_260 to 8,
            4_628 to 7, 13_639 to 21,
        )
        for ((frames, windows) in reference) {
            assertEquals("frames=$frames", windows, AnimeChordWindowPlan.windows(frames).size)
        }
    }

    @Test
    fun theFallbackGeometryAlsoCoversEveryFrame() {
        // `w15 o0` is the report's fallback if 455 MB of arena is too much: 646 frames,
        // no overlap. The same plan code has to serve it.
        for (frameCount in lengths) {
            val committed = mutableListOf<Int>()
            for (plan in AnimeChordWindowPlan.windows(frameCount, window = 646, overlap = 0)) {
                committed += plan.committedFrames.toList()
            }
            assertEquals("frameCount=$frameCount", (0 until frameCount).toList(), committed)
        }
    }

    @Test
    fun anEmptyFeatureMatrixProducesNoWindows() {
        assertTrue(AnimeChordWindowPlan.windows(0).isEmpty())
    }
}
