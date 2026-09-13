package com.nicos.pitchkit.tuner.harmony.lvchordia

import com.nicos.pitchkit.tuner.harmony.chordnet.CqtCpuFrontend
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtPlanDecoder

internal data class LvChordiaFeatures(
    val values: FloatArray,
    val frameCount: Int,
)

/** Android port of LV-Chordia's hybrid CQT input cropped to 252 model bins. */
internal class LvChordiaHybridCqtFrontend(
    lowPlanBytes: ByteArray,
    highPlanBytes: ByteArray,
) {
    private val recursive: CqtCpuFrontend
    private val pseudo: LvChordiaPseudoCqtFrontend

    init {
        val lowPlan = CqtPlanDecoder.decodeAndVerify(lowPlanBytes)
        require(lowPlan.config.sampleRate == LvChordiaContract.SAMPLE_RATE.toDouble())
        require(lowPlan.config.hopLength == LvChordiaContract.HOP_LENGTH)
        require(lowPlan.config.nBins == LvChordiaContract.ORIGINAL_RECURSIVE_BINS)
        require(lowPlan.config.binsPerOctave == 36)
        require(!lowPlan.config.logMagnitude)
        recursive = CqtCpuFrontend(lowPlan)

        val highPlan = LvChordiaPseudoPlanDecoder.decodeAndVerify(highPlanBytes)
        pseudo = LvChordiaPseudoCqtFrontend(highPlan)
    }

    fun transform(audio: FloatArray): LvChordiaFeatures {
        if (audio.isEmpty()) return LvChordiaFeatures(FloatArray(0), 0)
        val low = recursive.transform(audio)
        val high = pseudo.transform(audio)
        val frames = minOf(low.frameCount, high.frameCount)
        if (frames <= 0) return LvChordiaFeatures(FloatArray(0), 0)

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
        return LvChordiaFeatures(output, frames)
    }
}
