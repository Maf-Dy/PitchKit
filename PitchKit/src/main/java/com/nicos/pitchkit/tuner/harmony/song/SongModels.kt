package com.nicos.pitchkit.tuner.harmony.song

/** A stable chord interval in an analyzed song. */
data class SongChordSegment(
    val label: String,
    val startMs: Long,
    val endMs: Long,
    val confidence: Double,
)

/** A coarse repeated harmonic section. Labels intentionally stay A/B/C. */
data class SongSection(
    val label: String,
    val startMs: Long,
    val endMs: Long,
)

data class SongHarmonyAnalysis(
    val durationMs: Long,
    val chords: List<SongChordSegment>,
    val sections: List<SongSection>,
)
