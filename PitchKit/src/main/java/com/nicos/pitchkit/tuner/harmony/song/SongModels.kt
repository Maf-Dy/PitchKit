package com.nicos.pitchkit.tuner.harmony.song

data class SongChordAlternative(
    val label: String,
    val confidence: Double,
)

/**
 * A stable chord interval in an analyzed song.
 *
 * Optional harmonic details are retained so a later learning-oriented UI can
 * render piano/guitar/ukulele views without re-running the audio analysis.
 * Engines that only know a chord label can leave these fields empty.
 */
data class SongChordSegment(
    val label: String,
    val startMs: Long,
    val endMs: Long,
    val confidence: Double,
    val root: String? = null,
    val bass: String? = null,
    val pitchClasses: List<String> = emptyList(),
    val alternatives: List<SongChordAlternative> = emptyList(),
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
