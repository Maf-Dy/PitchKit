package com.nicos.pitchkit.tuner

import org.junit.Assert.assertEquals
import org.junit.Test

class FrameDropCounterTest {

    @Test
    fun `starts empty and reports no drop rate`() {
        val stats = FrameDropCounter().snapshot()
        assertEquals(0L, stats.delivered)
        assertEquals(0L, stats.processed)
        assertEquals(0L, stats.dropped)
        assertEquals(0.0, stats.dropRate, 0.0)
    }

    @Test
    fun `a fully drained run drops nothing`() {
        val counter = FrameDropCounter()
        repeat(10) {
            counter.onDelivered()
            counter.onProcessed()
        }
        val stats = counter.snapshot()
        assertEquals(10L, stats.delivered)
        assertEquals(10L, stats.processed)
        assertEquals(0L, stats.dropped)
        assertEquals(0.0, stats.dropRate, 0.0)
    }

    @Test
    fun `a slow inference leaves delivered minus processed as drops`() {
        val counter = FrameDropCounter()
        // Four buffers arrive while one inference runs: the conflated channel keeps
        // the newest and discards the three in between.
        repeat(4) { counter.onDelivered() }
        repeat(3) { counter.onDropped() }
        counter.onProcessed()

        val stats = counter.snapshot()
        assertEquals(4L, stats.delivered)
        assertEquals(3L, stats.dropped)
        assertEquals(1L, stats.processed)
        assertEquals(0.75, stats.dropRate, 1e-9)
        assertEquals(3L, counter.droppedCount())
    }

    @Test
    fun `a frame still in the channel is not yet a drop`() {
        val counter = FrameDropCounter()
        counter.onDelivered()
        assertEquals(0L, counter.droppedCount())
        assertEquals(0L, counter.snapshot().processed)
    }

    @Test
    fun `onDropped returns the running total so the caller can rate-limit its logging`() {
        val counter = FrameDropCounter()
        assertEquals(1L, counter.onDropped())
        assertEquals(2L, counter.onDropped())
    }

    @Test
    fun `reset clears every counter so a restarted flow counts from zero`() {
        val counter = FrameDropCounter()
        counter.onDelivered()
        counter.onDropped()
        counter.onProcessed()
        counter.reset()
        assertEquals(FrameDropStats(0L, 0L, 0L), counter.snapshot())
    }

    @Test
    fun `describe is a short debug line`() {
        assertEquals(
            "3 dropped of 4 delivered, 1 processed (75%)",
            FrameDropStats(delivered = 4L, processed = 1L, dropped = 3L).describe(),
        )
    }
}
