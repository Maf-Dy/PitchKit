package com.nicos.pitchkit.tuner.harmony.solitito

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The gate, the window fill, the vote and the latch, each on its own.
 *
 * `SolititoParityTest` proves the whole stack reproduces the Python reference on one real
 * clip; this proves each rule separately, on stimuli chosen to sit at its boundary, so a
 * regression says *which* rule moved.
 */
class SolititoDecisionLayerTest {

    private val features = FloatArray(SolititoContract.FEATURE_COUNT) { 0.5f }
    private val zero = FloatArray(SolititoContract.FEATURE_COUNT)

    /** A model that always answers the same thing, and counts how often it was asked. */
    private class Fixed(
        rootIndex: Int,
        quality: String,
        confidence: Double,
    ) : SolititoDecisionLayer.Model {
        private val reading = SolititoDecisionLayer.Reading(rootIndex, quality, confidence)
        var calls = 0
            private set

        override fun predict(window: FloatArray): SolititoDecisionLayer.Reading {
            calls++
            return reading
        }
    }

    /** A model whose answer the test changes between frames. */
    private class Programmable(
        var reading: SolititoDecisionLayer.Reading,
    ) : SolititoDecisionLayer.Model {
        var calls = 0
            private set

        override fun predict(window: FloatArray): SolititoDecisionLayer.Reading {
            calls++
            return reading
        }
    }

    // ------------------------------------------------------------------- the gate

    @Test
    fun theGateIsMinusSixtyDbfsOnTheRawFrameRms() {
        assertEquals(-60.0, SolititoContract.GATE_DB, 1e-12)
        assertEquals(-34.0, SolititoContract.SHIPPED_GATE_DB, 1e-12)
        assertEquals(0.001, SolititoContract.dbToLinear(SolititoContract.GATE_DB), 1e-12)
        assertEquals(
            0.0199526231496888,
            SolititoContract.dbToLinear(SolititoContract.SHIPPED_GATE_DB),
            1e-12,
        )

        val plan = SolititoDspPlanDecoder.decodeAndVerify(SolititoTestAssets.plan())
        val open = SolititoFrontend(plan, gateDb = SolititoContract.GATE_DB)
        val shipped = SolititoFrontend(plan, gateDb = SolititoContract.SHIPPED_GATE_DB)

        // Strictly greater-than, on the raw RMS, before the input gain.
        assertFalse(open.isLive(0.001f))
        assertTrue(open.isLive(0.0011f))
        // The 26 dB the port opens the gate by. A sustained chord decayed to -50 dBFS is
        // still analysed here and is thrown away upstream; that one constant was worth
        // +15.1 points of family accuracy on real guitar and -20.5 points of blank screen.
        val decayed = SolititoContract.dbToLinear(-50.0).toFloat()
        assertTrue("-50 dBFS must pass the port's gate", open.isLive(decayed))
        assertFalse("-50 dBFS must fail upstream's gate", shipped.isLive(decayed))
    }

    @Test
    fun rmsIsMeasuredOnTheRawFrameNotTheGainedOne() {
        val plan = SolititoDspPlanDecoder.decodeAndVerify(SolititoTestAssets.plan())
        val frontend = SolititoFrontend(plan)
        // A constant 0.25 frame has RMS 0.25; the x2.0 input gain would make it 0.5.
        val frame = FloatArray(SolititoContract.FFT_SIZE) { 0.25f }
        assertEquals(0.25, frontend.frameRms(frame, 0).toDouble(), 1e-6)
    }

    // ---------------------------------------------------------------- window fill

    @Test
    fun theModelIsNotAskedUntilHalfTheWindowIsLiveAndNamesNothingBelowNinetyPercent() {
        val model = Fixed(0, "maj", 0.9)
        val layer = SolititoDecisionLayer(model)
        val ticks = mutableListOf<Pair<Double, String?>>()
        // A full window of gated frames, then live ones: the fill walks 0 -> 1 one frame
        // at a time and every threshold is crossed exactly once.
        repeat(SolititoContract.CONTEXT_FRAMES) {
            layer.push(zero, false, 0f)?.let { ticks += it.fill to it.emitted }
        }
        repeat(SolititoContract.CONTEXT_FRAMES * 2) {
            layer.push(features, true, 0.5f)?.let { ticks += it.fill to it.emitted }
        }
        assertTrue(ticks.isNotEmpty())
        assertTrue("Every threshold must be crossed", ticks.any { it.first < SolititoContract.MIN_FILL })
        assertTrue(ticks.any { it.second == "N" })
        assertTrue(ticks.any { it.second == "C" })
        for ((fill, label) in ticks) {
            when {
                fill < SolititoContract.MIN_FILL ->
                    assertNull("fill $fill must not even ask", label)
                fill < SolititoContract.MIN_FILL_CHORD ->
                    assertEquals("fill $fill must not name a chord", "N", label)
                else -> assertEquals("fill $fill must name the chord", "C", label)
            }
        }
        assertEquals(
            "The model must be asked exactly on the ticks above MIN_FILL",
            ticks.count { it.first >= SolititoContract.MIN_FILL },
            model.calls,
        )
        assertEquals(model.calls.toLong(), layer.inferences)
        assertEquals(
            ticks.count { it.first < SolititoContract.MIN_FILL }.toLong(),
            layer.skippedLowFill,
        )
        assertEquals(
            SolititoContract.CONTEXT_FRAMES.toLong(),
            layer.gatedFrames,
        )
    }

    @Test
    fun ticksAreTwentyFiveASecondAndCarryTheFrameIndex() {
        val layer = SolititoDecisionLayer(Fixed(0, "maj", 0.9))
        val ticks = mutableListOf<Long>()
        repeat(100) { layer.push(features, true, 0.5f)?.let { ticks += it.frameIndex } }
        assertEquals(listOf(0L, 2L, 5L, 7L, 10L, 12L, 15L, 17L), ticks.take(8))
        // 100 frames is 1.6 s at a 16 ms hop; 25 ticks a second is 40 of them.
        assertEquals(40, ticks.size)
    }

    // ---------------------------------------------------------------------- vote

    @Test
    fun theVoteIsConfidenceWeightedOverThreeWindowsAndBreaksTiesByFirstSeen() {
        // Warm the window to fully live while the model says nothing at all, so the vote
        // starts from three `Noise` entries whose confidence is 0.0 and which therefore
        // cannot win anything.
        val model = Programmable(SolititoDecisionLayer.Reading(12, "maj", 0.0))
        val layer = SolititoDecisionLayer(model)
        val warm = drive(layer, frames = 120)
        assertEquals("Noise must leave the screen empty", "N", warm.last())

        // A tie is broken by the first chord inserted: `Noise` scores 0.0, then C and
        // A m7 both score 0.50, and C was seen first.
        model.reading = SolititoDecisionLayer.Reading(0, "maj", 0.50)
        assertEquals("C", tick(layer))
        model.reading = SolititoDecisionLayer.Reading(9, "min7", 0.50)
        assertEquals("C", tick(layer))

        // Once A m7 has two of the three windows it leads 1.00 to 0.50 and takes over.
        assertEquals("Am7", tick(layer))

        // ...and two windows of a *quiet* A m7 still outweigh one confident C, which is
        // the whole point of weighting the vote rather than counting it.
        model.reading = SolititoDecisionLayer.Reading(9, "min7", 0.30)
        tick(layer)
        tick(layer)
        model.reading = SolititoDecisionLayer.Reading(0, "maj", 0.50)
        assertEquals("Am7", tick(layer))
        // The third C finally wins it: 1.00 against 0.30.
        assertEquals("C", tick(layer))
        assertEquals("C", tick(layer))
    }

    @Test
    fun aWindowOfNothingButNoiseNeverResolvesToALabel() {
        // `Noise` carries confidence 0.0 upstream, so it can never beat the 0.0 the tally
        // starts at; the screen stays empty rather than showing a phantom chord.
        val labels = drive(SolititoDecisionLayer(Fixed(12, "maj", 0.0)), frames = 120)
        assertTrue(labels.isNotEmpty())
        assertTrue("Noise must never name a chord: $labels", labels.all { it == "N" })
    }

    @Test
    fun aSingleNoteIsNotAChord() {
        // 99.3 % of a rootless piano voicing comes back as `note`. It is a refusal, not a
        // chord, and it must not reach the screen as one.
        val labels = drive(SolititoDecisionLayer(Fixed(0, "note", 0.95)), frames = 120)
        assertTrue("`note` must show nothing: $labels", labels.all { it == "N" })
    }

    // --------------------------------------------------------------------- latch

    @Test
    fun theLatchHoldsAFourNoteChordThroughItsDecayAndReleasesOnANewAttack() {
        val latch = SolititoChordLatch(enabled = true)
        // Nothing locks before the chord has settled.
        assertEquals("G m7", latch.update(1, 10, "G m7", 0.97))
        assertNull(latch.locked)

        assertEquals("G m7", latch.update(1, SolititoContract.SETTLE_FRAMES, "G m7", 0.95))
        assertEquals("G m7", latch.locked)
        // The seventh dies away and the model hears the triad: the label does not move.
        assertEquals("G m7", latch.update(1, SolititoContract.SETTLE_FRAMES + 10, "G m", 0.99))
        // ...nor when only one note of it is left ringing.
        assertEquals("G m7", latch.update(1, SolititoContract.SETTLE_FRAMES + 12, "Note G", 0.99))
        // A confident reading of a genuinely different chord still gets through at once.
        assertEquals("G m7b5", latch.update(1, SolititoContract.SETTLE_FRAMES + 20, "G m7b5", 0.93))
        assertEquals("G m7b5", latch.locked)
        // An unconfident one does not.
        assertEquals("G m7b5", latch.update(1, SolititoContract.SETTLE_FRAMES + 22, "C", 0.10))
        // A different root is not a decay, so it is held rather than followed while quiet.
        assertEquals("G m7b5", latch.update(1, SolititoContract.SETTLE_FRAMES + 24, "C m", 0.10))
        // A new attack releases the lock outright.
        assertEquals("C", latch.update(2, SolititoContract.SETTLE_FRAMES, "C", 0.95))
    }

    @Test
    fun theLatchIsOffWhenDisabled() {
        val latch = SolititoChordLatch(enabled = false)
        latch.update(1, SolititoContract.SETTLE_FRAMES, "G m7", 0.99)
        assertNull(latch.locked)
        assertEquals("G m", latch.update(1, SolititoContract.SETTLE_FRAMES + 10, "G m", 0.99))
    }

    @Test
    fun theAttackDetectorSegmentsStrikesWithARefractoryPeriod() {
        val layer = SolititoDecisionLayer(Fixed(0, "maj7", 0.95))
        val onsets = mutableSetOf<Int>()
        // Below the 0.01 attack floor: nothing fires however long it runs.
        repeat(30) { layer.push(features, true, 0.002f)?.let { onsets += it.onsetId } }
        // A strike. The envelope's 0.90/0.10 rise then keeps it from firing again while
        // the note is held, which is what the 12-frame refractory and the 1.8 ratio are
        // between them for.
        repeat(60) { layer.push(features, true, 0.40f)?.let { onsets += it.onsetId } }
        // Decay, then a second strike.
        repeat(60) { layer.push(features, true, 0.005f)?.let { onsets += it.onsetId } }
        repeat(30) { layer.push(features, true, 0.40f)?.let { onsets += it.onsetId } }
        assertEquals("Two strikes, two onsets, numbered from zero", setOf(0, 1, 2), onsets)
    }

    @Test
    fun resetClearsEverythingTheNextTakeWouldOtherwiseInherit() {
        val layer = SolititoDecisionLayer(Fixed(0, "maj", 0.95))
        val before = drive(layer, frames = 150)
        assertTrue(layer.inferences > 0)
        assertTrue(before.contains("C"))

        layer.reset()
        assertEquals(0L, layer.inferences)
        assertEquals(0L, layer.liveFrames)
        assertEquals(0L, layer.skippedLowFill)

        // The frame counter restarts, so the tick schedule does...
        val resumed = mutableListOf<Long>()
        repeat(10) { layer.push(features, true, 0.5f)?.let { resumed += it.frameIndex } }
        assertEquals(listOf(0L, 2L, 5L, 7L), resumed)
        // ...and so does the history: a window this cold cannot even be asked about.
        assertNull(after(layer, 0))
    }

    // ------------------------------------------------------------------- helpers

    /** The label the first tick at or after this point emitted. */
    private fun tick(layer: SolititoDecisionLayer): String? {
        repeat(SolititoContract.CONTEXT_FRAMES * 4) {
            val decision = layer.push(features, true, 0.5f)
            if (decision != null) return decision.emitted
        }
        throw AssertionError("No tick in four windows")
    }

    /** The emitted label of the tick [skip] ticks from now, or null if it named nothing. */
    private fun after(layer: SolititoDecisionLayer, skip: Int): String? {
        var seen = 0
        repeat(SolititoContract.CONTEXT_FRAMES * 4) {
            val decision = layer.push(features, true, 0.5f)
            if (decision != null) {
                if (seen == skip) return decision.emitted
                seen++
            }
        }
        throw AssertionError("No tick in four windows")
    }

    /** Feed [frames] live frames and return the labels the ticks actually emitted. */
    private fun drive(layer: SolititoDecisionLayer, frames: Int): List<String> {
        val labels = mutableListOf<String>()
        repeat(frames) {
            layer.push(features, true, 0.5f)?.emitted?.let { labels += it }
        }
        return labels
    }
}
