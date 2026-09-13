package com.nicos.pitchkit.tuner.harmony.crema

import com.nicos.pitchkit.tuner.harmony.chordnet.CqtCpuFrontend
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtPlanDecoder
import java.security.MessageDigest
import java.util.concurrent.Executors
import java.util.concurrent.ThreadFactory
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.log10
import kotlin.math.max

internal data class CremaFeatures(
    val values: FloatArray,
    val frameCount: Int,
)

/** Builds Crema 0.2.0's two-channel HCQT feature tensor. */
internal class CremaHcqtFrontend(
    harmonic1PlanBytes: ByteArray,
    harmonic2PlanBytes: ByteArray,
) : AutoCloseable {
    private val harmonic1: CqtCpuFrontend
    private val harmonic2: CqtCpuFrontend
    private val executor = Executors.newFixedThreadPool(2, CremaThreadFactory())

    init {
        val h1Hash = sha256(harmonic1PlanBytes)
        val h2Hash = sha256(harmonic2PlanBytes)
        require(h1Hash == CremaContract.PLAN_H1_SHA256) {
            "Unexpected Crema harmonic-1 CQT plan: size=${harmonic1PlanBytes.size}, sha256=$h1Hash"
        }
        require(h2Hash == CremaContract.PLAN_H2_SHA256) {
            "Unexpected Crema harmonic-2 CQT plan: size=${harmonic2PlanBytes.size}, sha256=$h2Hash"
        }

        val plan1 = CqtPlanDecoder.decodeAndVerify(harmonic1PlanBytes)
        val plan2 = CqtPlanDecoder.decodeAndVerify(harmonic2PlanBytes)
        validatePlan(plan1.config.sampleRate, plan1.config.hopLength, plan1.config.nBins, plan1.config.logMagnitude)
        validatePlan(plan2.config.sampleRate, plan2.config.hopLength, plan2.config.nBins, plan2.config.logMagnitude)
        harmonic1 = CqtCpuFrontend(plan1)
        harmonic2 = CqtCpuFrontend(plan2)
    }

    fun transform(audio: FloatArray): CremaFeatures {
        val firstFuture = executor.submit<CqtCpuFrontend.Features> { harmonic1.transform(audio) }
        val secondFuture = executor.submit<CqtCpuFrontend.Features> { harmonic2.transform(audio) }

        val first = firstFuture.get()
        val second = secondFuture.get()
        val frames = minOf(first.frameCount, second.frameCount)
        require(frames > 0)
        require(first.binCount == CremaContract.INPUT_BINS)
        require(second.binCount == CremaContract.INPUT_BINS)

        val expectedFrames = (audio.size / CremaContract.HOP_LENGTH).coerceAtLeast(1)
        val usableFrames = minOf(frames, expectedFrames)

        val firstDb = amplitudeToDb(first.values, usableFrames, first.binCount)
        val secondDb = amplitudeToDb(second.values, usableFrames, second.binCount)
        val interleaved = FloatArray(
            usableFrames * CremaContract.INPUT_BINS * CremaContract.HARMONIC_CHANNELS
        )

        for (frame in 0 until usableFrames) {
            for (bin in 0 until CremaContract.INPUT_BINS) {
                val source = frame * CremaContract.INPUT_BINS + bin
                val target = source * CremaContract.HARMONIC_CHANNELS
                interleaved[target] = firstDb[source]
                interleaved[target + 1] = secondDb[source]
            }
        }
        return CremaFeatures(interleaved, usableFrames)
    }

    override fun close() {
        executor.shutdownNow()
    }

    private fun amplitudeToDb(values: FloatArray, frames: Int, bins: Int): FloatArray {
        val count = frames * bins
        var reference = 1e-5
        for (index in 0 until count) reference = max(reference, values[index].toDouble())
        val referenceDb = 20.0 * log10(reference)
        val result = FloatArray(count)
        for (index in 0 until count) {
            val magnitude = max(1e-5, values[index].toDouble())
            val db = 20.0 * log10(magnitude) - referenceDb
            result[index] = db.coerceAtLeast(-80.0).toFloat()
        }
        return result
    }

    private fun validatePlan(sampleRate: Double, hopLength: Int, bins: Int, logMagnitude: Boolean) {
        require(sampleRate == CremaContract.SAMPLE_RATE.toDouble())
        require(hopLength == CremaContract.HOP_LENGTH)
        require(bins == CremaContract.INPUT_BINS)
        require(!logMagnitude) { "Crema CQT plans must output linear magnitude" }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private class CremaThreadFactory : ThreadFactory {
        private val counter = AtomicInteger(1)

        override fun newThread(runnable: Runnable): Thread = Thread(
            runnable,
            "Crema-HCQT-${counter.getAndIncrement()}",
        ).apply {
            isDaemon = true
        }
    }
}
