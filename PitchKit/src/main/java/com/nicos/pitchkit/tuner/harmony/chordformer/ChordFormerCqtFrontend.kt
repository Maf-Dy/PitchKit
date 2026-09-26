package com.nicos.pitchkit.tuner.harmony.chordformer

import com.nicos.pitchkit.tuner.harmony.chordnet.CqtCpuFrontend
import com.nicos.pitchkit.tuner.harmony.chordnet.CqtPlanDecoder
import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaPseudoCqtFrontend
import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaPseudoPlanDecoder
import kotlin.math.log10
import kotlin.math.max

internal data class ChordFormerFeatures(
    /** Row-major [frameCount x INPUT_BINS] decibel values in -80..0. */
    val values: FloatArray,
    val frameCount: Int,
)

/**
 * Android port of ChordFormer's `extractors/cqt.py::CQTV2`.
 *
 * The magnitudes come from the same plan-driven hybrid CQT LV Song uses (a recursive
 * librosa CQT for the low 238 bins plus a pseudo-CQT for the top 50). ChordFormer then
 * applies `librosa.amplitude_to_db(magnitude, ref=np.max)` over the *whole* 288-bin
 * spectrogram and only afterwards crops to the 252 model bins, so the dB reference is a
 * property of the whole song and the frontend has to see every bin — which is why
 * ChordFormer ships its own pseudo plan keeping all 50 pseudo bins.
 *
 * Known deviation from upstream: `CQTV2` passes `tuning=None`, so librosa re-estimates the
 * tuning of each recording and shifts fmin by up to ±1/6 semitone. The plan is baked at
 * tuning=0, exactly as the LV Song port already is; the app's own reference-pitch control
 * is what moves the grid here.
 */
internal class ChordFormerCqtFrontend(
    lowPlanBytes: ByteArray,
    highPlanBytes: ByteArray,
) {
    private val recursive: CqtCpuFrontend
    private val pseudo: LvChordiaPseudoCqtFrontend

    init {
        val lowPlan = CqtPlanDecoder.decodeAndVerify(lowPlanBytes)
        require(lowPlan.config.sampleRate == ChordFormerContract.SAMPLE_RATE.toDouble())
        require(lowPlan.config.hopLength == ChordFormerContract.HOP_LENGTH)
        require(lowPlan.config.nBins == ChordFormerContract.RECURSIVE_BINS)
        require(lowPlan.config.binsPerOctave == ChordFormerContract.BINS_PER_OCTAVE)
        require(!lowPlan.config.logMagnitude) {
            "ChordFormer applies its own dB scaling; the plan must emit linear magnitudes"
        }
        recursive = CqtCpuFrontend(lowPlan)

        val highPlan = LvChordiaPseudoPlanDecoder.decodeAndVerify(
            highPlanBytes,
            expectedBinCount = ChordFormerContract.PSEUDO_BINS,
        )
        pseudo = LvChordiaPseudoCqtFrontend(highPlan)
    }

    fun transform(audio: FloatArray): ChordFormerFeatures {
        if (audio.isEmpty()) return ChordFormerFeatures(FloatArray(0), 0)
        val low = recursive.transform(audio)
        val high = pseudo.transform(audio)
        val frames = minOf(low.frameCount, high.frameCount)
        if (frames <= 0) return ChordFormerFeatures(FloatArray(0), 0)

        // Pass 1: the dB reference is the loudest bin of the whole 288-bin spectrogram,
        // including the 18 low and 18 high bins the model never sees.
        var peak = 0.0
        for (frame in 0 until frames) {
            val lowOffset = frame * low.binCount
            for (bin in 0 until ChordFormerContract.RECURSIVE_BINS) {
                val value = low.values[lowOffset + bin].toDouble()
                if (value > peak) peak = value
            }
            val highOffset = frame * high.binCount
            for (bin in 0 until ChordFormerContract.PSEUDO_BINS) {
                val value = high.values[highOffset + bin].toDouble()
                if (value > peak) peak = value
            }
        }
        val referenceDb = 20.0 * log10(max(ChordFormerContract.AMPLITUDE_FLOOR, peak))
        val floorDb = -ChordFormerContract.TOP_DB

        // Pass 2: convert and crop to bins 18..270 in one walk.
        val output = FloatArray(frames * ChordFormerContract.INPUT_BINS)
        for (frame in 0 until frames) {
            val destination = frame * ChordFormerContract.INPUT_BINS
            val lowOffset = frame * low.binCount
            val highOffset = frame * high.binCount
            for (index in 0 until ChordFormerContract.INPUT_BINS) {
                val hybridBin = index + ChordFormerContract.DROP_LOW_BINS
                val magnitude = if (hybridBin < ChordFormerContract.RECURSIVE_BINS) {
                    low.values[lowOffset + hybridBin].toDouble()
                } else {
                    high.values[highOffset + hybridBin - ChordFormerContract.RECURSIVE_BINS].toDouble()
                }
                val decibels =
                    20.0 * log10(max(ChordFormerContract.AMPLITUDE_FLOOR, magnitude)) - referenceDb
                output[destination + index] = max(floorDb, decibels).toFloat()
            }
        }

        return ChordFormerFeatures(values = output, frameCount = frames)
    }
}
