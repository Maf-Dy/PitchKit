package com.nicos.pitchkit.tuner.harmony.lvchordia

import com.nicos.pitchkit.tuner.harmony.chordnet.CqtCpuFrontend
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtPlanDecoder
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

internal data class LvChordiaFeatures(
    val values: FloatArray,
    val frameCount: Int,
    /** Full-register normalized 12-bin pitch-class evidence. */
    val chroma: FloatArray,
    /** Mid/upper-register evidence used for chord quality and extensions. */
    val harmonyChroma: FloatArray,
    /** Low-register evidence used for bass/root interpretation. */
    val bassChroma: FloatArray,
)

/** Android port of LV Song's hybrid CQT input cropped to 252 model bins. */
internal class LvChordiaHybridCqtFrontend(
    lowPlanBytes: ByteArray,
    highPlanBytes: ByteArray,
) {
    private companion object {
        const val HARMONY_MIN_HZ = 130.0
        const val HARMONY_MAX_HZ = 3000.0
        const val BASS_MAX_HZ = 330.0
    }

    private val recursive: CqtCpuFrontend
    private val pseudo: LvChordiaPseudoCqtFrontend
    private val recursiveFmin: Double
    private val recursiveBinsPerOctave: Int

    init {
        val lowPlan = CqtPlanDecoder.decodeAndVerify(lowPlanBytes)
        require(lowPlan.config.sampleRate == LvChordiaContract.SAMPLE_RATE.toDouble())
        require(lowPlan.config.hopLength == LvChordiaContract.HOP_LENGTH)
        require(lowPlan.config.nBins == LvChordiaContract.ORIGINAL_RECURSIVE_BINS)
        require(lowPlan.config.binsPerOctave == 36)
        require(!lowPlan.config.logMagnitude)
        recursive = CqtCpuFrontend(lowPlan)
        recursiveFmin = lowPlan.config.fmin
        recursiveBinsPerOctave = lowPlan.config.binsPerOctave

        val highPlan = LvChordiaPseudoPlanDecoder.decodeAndVerify(highPlanBytes)
        pseudo = LvChordiaPseudoCqtFrontend(highPlan)
    }

    fun transform(audio: FloatArray): LvChordiaFeatures {
        if (audio.isEmpty()) return emptyFeatures()
        val low = recursive.transform(audio)
        val high = pseudo.transform(audio)
        val frames = minOf(low.frameCount, high.frameCount)
        if (frames <= 0) return emptyFeatures()

        val output = FloatArray(frames * LvChordiaContract.INPUT_BINS)
        for (frame in 0 until frames) {
            val outputOffset = frame * LvChordiaContract.INPUT_BINS
            val lowOffset = frame * low.binCount + LvChordiaContract.DROP_LOW_BINS
            low.values.copyInto(
                destination = output,
                destinationOffset = outputOffset,
                startIndex = lowOffset,
                endIndex = lowOffset + LvChordiaContract.RECURSIVE_MODEL_BINS,
            )

            val highOffset = frame * high.binCount
            high.values.copyInto(
                destination = output,
                destinationOffset = outputOffset + LvChordiaContract.RECURSIVE_MODEL_BINS,
                startIndex = highOffset,
                endIndex = highOffset + LvChordiaContract.PSEUDO_MODEL_BINS,
            )
        }

        return LvChordiaFeatures(
            values = output,
            frameCount = frames,
            chroma = foldRecursiveCqtToChroma(
                low = low,
                frames = frames,
                minFrequencyHz = 0.0,
                maxFrequencyHz = Double.POSITIVE_INFINITY,
                bassWeighting = false,
            ),
            harmonyChroma = foldRecursiveCqtToChroma(
                low = low,
                frames = frames,
                minFrequencyHz = HARMONY_MIN_HZ,
                maxFrequencyHz = HARMONY_MAX_HZ,
                bassWeighting = false,
            ),
            bassChroma = foldRecursiveCqtToChroma(
                low = low,
                frames = frames,
                minFrequencyHz = 0.0,
                maxFrequencyHz = BASS_MAX_HZ,
                bassWeighting = true,
            ),
        )
    }

    private fun emptyFeatures() = LvChordiaFeatures(
        values = FloatArray(0),
        frameCount = 0,
        chroma = FloatArray(0),
        harmonyChroma = FloatArray(0),
        bassChroma = FloatArray(0),
    )

    private fun foldRecursiveCqtToChroma(
        low: CqtCpuFrontend.Features,
        frames: Int,
        minFrequencyHz: Double,
        maxFrequencyHz: Double,
        bassWeighting: Boolean,
    ): FloatArray {
        val output = FloatArray(frames * 12)
        for (frame in 0 until frames) {
            val row = frame * 12
            val source = frame * low.binCount
            for (bin in 0 until low.binCount) {
                val frequency = recursiveFmin * 2.0.pow(
                    bin.toDouble() / recursiveBinsPerOctave.toDouble()
                )
                if (frequency < minFrequencyHz || frequency > maxFrequencyHz) continue

                val midi = (
                    69.0 + 12.0 * (ln(frequency / 440.0) / ln(2.0))
                ).roundToInt()
                val pitchClass = ((midi % 12) + 12) % 12
                val magnitude = low.values[source + bin].toDouble().coerceAtLeast(0.0)
                val compressed = sqrt(magnitude)
                val registerWeight = if (bassWeighting) {
                    sqrt((recursiveFmin / frequency).coerceAtMost(1.0))
                } else {
                    1.0
                }
                output[row + pitchClass] += (compressed * registerWeight).toFloat()
            }

            var peak = 0f
            for (pc in 0 until 12) peak = maxOf(peak, output[row + pc])
            if (peak > 1e-8f) {
                for (pc in 0 until 12) output[row + pc] /= peak
            }
        }
        return output
    }
}
