package com.nicos.pitchkit.tuner.harmony.chordnet

data class CqtConfig(
    val sampleRate: Double,
    val hopLength: Int,
    val nBins: Int,
    val binsPerOctave: Int,
    val fmin: Double,
    val logMagnitude: Boolean,
)

data class CqtOctavePlan(
    val index: Int,
    val sampleRate: Double,
    val hopLength: Int,
    val fftSize: Int,
    val binStart: Int,
    val binCount: Int,
)

data class CqtDownsamplePlan(
    val tapCount: Int,
    val halfCoefficients: FloatArray,
    val gain: Double = kotlin.math.sqrt(2.0),
    val delay: Int = (tapCount - 1) / 2,
)

data class CqtPlan(
    val formatVersion: Int,
    val generator: String,
    val config: CqtConfig,
    val earlyDownsampleCount: Int,
    val octaves: List<CqtOctavePlan>,
    val rowOffsets: IntArray,
    val fftBins: IntArray,
    val coefficients: FloatArray,
    val binLengths: FloatArray,
    val downsample: CqtDownsamplePlan,
    val payloadSha256: String,
)
