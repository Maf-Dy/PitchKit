package com.nicos.pitchkit.tuner

import java.util.concurrent.atomic.AtomicLong
import kotlin.math.roundToInt

/**
 * Counts the capture buffers that the conflated capture → processor channel throws away.
 *
 * [TunerEngine] hands frames to a `Channel.CONFLATED`, which is a drop, not
 * backpressure: when one inference takes longer than one capture period the
 * intermediate buffers are discarded and the recognizer's ring buffer receives
 * spliced audio. Nothing counted that before, so the accuracy cost of a slow
 * device was invisible. This counts it and changes nothing else — the channel
 * keeps its current capacity and overflow behaviour.
 *
 * Safe to touch from the capture thread and the processor coroutine at once.
 */
internal class FrameDropCounter {
    private val delivered = AtomicLong()
    private val processed = AtomicLong()
    private val dropped = AtomicLong()

    /** A capture buffer was accepted by the channel. */
    fun onDelivered() {
        delivered.incrementAndGet()
    }

    /** The processor took a buffer off the channel. */
    fun onProcessed() {
        processed.incrementAndGet()
    }

    /** The channel discarded a buffer before anyone processed it; returns the new total. */
    fun onDropped(): Long = dropped.incrementAndGet()

    fun reset() {
        delivered.set(0)
        processed.set(0)
        dropped.set(0)
    }

    /** Cheap enough to read on every emitted frame; that is the whole point of it. */
    fun droppedCount(): Long = dropped.get()

    /**
     * The three counters. They are read independently, so a snapshot taken while
     * audio is flowing can be off by the frame in flight; it is a diagnostic, not
     * a ledger.
     */
    fun snapshot(): FrameDropStats = FrameDropStats(
        delivered = delivered.get(),
        processed = processed.get(),
        dropped = dropped.get(),
    )
}

internal data class FrameDropStats(
    val delivered: Long,
    val processed: Long,
    val dropped: Long,
) {
    /** Fraction of delivered buffers the channel threw away; 0.0 before any audio arrives. */
    val dropRate: Double get() = if (delivered <= 0L) 0.0 else dropped.toDouble() / delivered

    /** One short line for a debug log or the debug field on the live screen. */
    fun describe(): String =
        "$dropped dropped of $delivered delivered, $processed processed (${(dropRate * 100).roundToInt()}%)"
}
