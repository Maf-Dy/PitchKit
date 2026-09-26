package com.nicos.pitchkit.tuner.harmony.solitito

import kotlin.math.abs
import kotlin.math.ln
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * The CSR kernel loader, against the facts the upstream generator
 * (`dist/gen_weights.py`) and the Stage C part 2 screening establish about the asset.
 *
 * The shipped `solitito-dsp.bin` is packed from upstream's `dsp_weights.json` by
 * `tools/accuracy_audit/export_solitito_assets.py`; nothing is re-derived on the way, so
 * anything this test finds is either a packing bug or a swapped asset.
 */
class SolititoDspPlanTest {

    private fun planBytesOrSkip(): ByteArray {
        val file = SolititoTestAssets.locate(SolititoContract.PLAN_FILE)
        assumeTrue(
            "solitito assets are not installed; run " +
                "tools/accuracy_audit/export_solitito_assets.py",
            file != null,
        )
        return file!!.readBytes()
    }

    @Test
    fun decodesTheShippedKernelWithTheGeometryTheContractPins() {
        val bytes = planBytesOrSkip()
        assertEquals(
            "solitito-dsp.bin does not match SolititoContract.PLAN_SHA256",
            SolititoContract.PLAN_SHA256,
            SolititoContract.sha256(bytes),
        )
        val plan = SolititoDspPlanDecoder.decodeAndVerify(bytes)

        assertEquals(1, plan.formatVersion)
        assertEquals(SolititoContract.SAMPLE_RATE, plan.sampleRate)
        assertEquals(SolititoContract.FFT_SIZE, plan.fftSize)
        assertEquals(SolititoContract.FFT_BINS, plan.fftBinCount)
        assertEquals(SolititoContract.CQT_BINS, plan.binCount)
        assertEquals(SolititoContract.CHROMA_BINS, plan.chromaBinCount)

        // `gen_weights.py` prunes at 1e-4 of peak and reports 6.9 % density.
        assertEquals("Upstream's surviving weight count", 40_675, plan.nonZeroCount)
        val density = plan.nonZeroCount.toDouble() / (plan.fftBinCount * plan.binCount)
        assertEquals("Kernel density", 0.069, density, 0.002)

        assertEquals(plan.binCount + 1, plan.rowOffsets.size)
        assertEquals(0, plan.rowOffsets.first())
        assertEquals(plan.nonZeroCount, plan.rowOffsets.last())
        assertEquals(plan.nonZeroCount, plan.fftBins.size)
        assertEquals(plan.nonZeroCount, plan.weightsReal.size)
        assertEquals(plan.nonZeroCount, plan.weightsImaginary.size)
        assertEquals(plan.binCount * plan.chromaBinCount, plan.chroma.size)

        var peak = 0.0
        var smallest = Double.MAX_VALUE
        for (index in 0 until plan.nonZeroCount) {
            val magnitude = Math.hypot(
                plan.weightsReal[index].toDouble(),
                plan.weightsImaginary[index].toDouble(),
            )
            if (magnitude > peak) peak = magnitude
            if (magnitude < smallest) smallest = magnitude
        }
        // Peak-normalised, pruned at 1e-4 of peak — both are upstream's documented claims
        // and both were re-measured by the screening.
        assertEquals("Kernel peak", 1.0, peak, 1e-6)
        assertTrue("Smallest kept weight $smallest", smallest >= 9e-5)
        assertTrue("Smallest kept weight $smallest", smallest <= 2e-4)

        // Every CQT bin must actually have taps, or a whole semitone reads as silence.
        for (bin in 0 until plan.binCount) {
            assertTrue(
                "CQT bin $bin has no kernel weights",
                plan.rowOffsets[bin + 1] > plan.rowOffsets[bin],
            )
        }
    }

    @Test
    fun kernelBinCentresAreTheQuarterToneGridFromC1() {
        val plan = SolititoDspPlanDecoder.decodeAndVerify(planBytesOrSkip())
        val binHz = plan.sampleRate.toDouble() / plan.fftSize
        val c1 = 32.703195662574829
        var worstCents = 0.0
        // Below C3 the 1.95 Hz FFT grid cannot resolve a quarter tone at all, which is a
        // property of the 8192-point window, not of the packing — so the grid is checked
        // where it is resolvable, exactly as the screening checked it.
        for (bin in 48 until plan.binCount) {
            var weighted = 0.0
            var total = 0.0
            for (index in plan.rowOffsets[bin] until plan.rowOffsets[bin + 1]) {
                val magnitude = Math.hypot(
                    plan.weightsReal[index].toDouble(),
                    plan.weightsImaginary[index].toDouble(),
                )
                weighted += plan.fftBins[index] * magnitude
                total += magnitude
            }
            val centreHz = (weighted / total) * binHz
            val expected = c1 * Math.pow(2.0, bin / 24.0)
            worstCents = maxOf(worstCents, abs(1200.0 * ln(centreHz / expected) / ln(2.0)))
        }
        println("solitito kernel: worst bin-centre error ${"%.2f".format(worstCents)} cents")
        assertTrue("Worst bin centre error $worstCents cents", worstCents <= 15.0)
    }

    @Test
    fun chromaFoldsEachCqtBinIntoExactlyOneClass() {
        val plan = SolititoDspPlanDecoder.decodeAndVerify(planBytesOrSkip())
        for (bin in 0 until plan.binCount) {
            var nonZero = 0
            var classIndex = -1
            for (chromaBin in 0 until plan.chromaBinCount) {
                if (plan.chroma[bin * plan.chromaBinCount + chromaBin] != 0f) {
                    nonZero++
                    classIndex = chromaBin
                }
            }
            assertEquals("CQT bin $bin folds into $nonZero chroma classes", 1, nonZero)
            // `cq_to_chroma` *rounds* the quarter-tone bin onto a semitone, so the mapping
            // is ((k + 1) / 2) % 12 and bin 0 sits alone in class C. Upstream's own
            // `gen_weights.py` compares against a stale `(k // 2) % 12` and prints a
            // frightening, false "train/serve mismatch" warning when they differ; they
            // always differ, the kernel is right, and the comparator is the bug.
            assertEquals("CQT bin $bin chroma class", ((bin + 1) / 2) % 12, classIndex)
        }
    }

    @Test
    fun rejectsATamperedArtifact() {
        val bytes = planBytesOrSkip()
        assertNotNull(SolititoDspPlanDecoder.decodeAndVerify(bytes))

        val flippedPayload = bytes.copyOf()
        flippedPayload[bytes.size - 1] = (flippedPayload[bytes.size - 1].toInt() xor 0x01).toByte()
        expectFailure("a flipped payload byte") {
            SolititoDspPlanDecoder.decodeAndVerify(flippedPayload)
        }

        val flippedMagic = bytes.copyOf()
        flippedMagic[0] = (flippedMagic[0].toInt() xor 0xff).toByte()
        expectFailure("a broken magic") { SolititoDspPlanDecoder.decodeAndVerify(flippedMagic) }

        val truncated = bytes.copyOfRange(0, bytes.size - 4)
        expectFailure("a truncated artifact") {
            SolititoDspPlanDecoder.decodeAndVerify(truncated)
        }

        expectFailure("an empty artifact") {
            SolititoDspPlanDecoder.decodeAndVerify(ByteArray(0))
        }
    }

    private inline fun expectFailure(what: String, block: () -> Unit) {
        try {
            block()
            fail("The decoder accepted $what")
        } catch (expected: IllegalArgumentException) {
            // as intended
        } catch (expected: IndexOutOfBoundsException) {
            // as intended: a truncated artifact can die in the buffer itself
        }
    }

    @Test
    fun inferenceCadenceIsTheFortyMillisecondTickSchedule() {
        // `int(step * 0.040 * 16000 / 256)` over the integers, which is what the
        // reference builds its tick set from.
        val period = SolititoContract.INFERENCE_PERIOD_SECONDS *
            SolititoContract.SAMPLE_RATE / SolititoContract.HOP_LENGTH
        assertEquals(2.5, period, 1e-12)
        val expected = mutableSetOf<Long>()
        for (step in 0 until 400) expected.add(Math.floor(step * period).toLong())
        for (frame in 0L until 1000L) {
            assertEquals(
                "frame $frame",
                frame in expected,
                SolititoContract.isInferenceFrame(frame),
            )
        }
        // 25 ticks a second: 62.5 frames a second, two ticks every five frames.
        assertEquals(25.0, 2.0 / 5 * SolititoContract.SAMPLE_RATE / SolititoContract.HOP_LENGTH, 1e-9)
    }
}
