package com.nicos.pitchkit.tuner.harmony.animechord

/** One decoded chord interval, still in the model's own frame time. */
internal data class AnimeChordEvent(
    val startSeconds: Double,
    val endSeconds: Double,
    val rootChordIndex: Int,
    val bassIndex: Int,
)

/**
 * `PredictionDecoder.decode_frames` + `to_events`, chord head only.
 *
 * The frame-level chord is the root-chord label with the bass head's slash appended —
 * dropped when the bass says `N` or repeats the root, which is what upstream does — and
 * events are the runs of that combined label. `min_duration_chord = 0.1 s` then folds a
 * run shorter than a tenth of a second into its predecessor, which can make the
 * predecessor's own predecessor adjacent to an identical label; upstream merges that pair
 * too, and so does this.
 *
 * The key head is not decoded. It feeds only the romanizer, which the reference lane runs
 * with `romanize=False` because `chord-romanizer` is not installed, and the app has no use
 * for it.
 */
internal object AnimeChordDecoder {
    fun events(
        rootPath: IntArray,
        bassPath: IntArray,
        vocabulary: AnimeChordVocabulary,
        secondsPerFrame: Double = AnimeChordContract.SECONDS_PER_FRAME,
        minChordSeconds: Double = AnimeChordContract.MIN_CHORD_SECONDS,
    ): List<AnimeChordEvent> {
        val frames = minOf(rootPath.size, bassPath.size)
        if (frames == 0) return emptyList()

        val raw = ArrayList<AnimeChordEvent>()
        var lastKey = vocabulary.chordKey(rootPath[0], bassPath[0])
        var startFrame = 0
        for (frame in 1 until frames) {
            val key = vocabulary.chordKey(rootPath[frame], bassPath[frame])
            if (key == lastKey) continue
            raw += AnimeChordEvent(
                startSeconds = startFrame * secondsPerFrame,
                endSeconds = frame * secondsPerFrame,
                rootChordIndex = rootPath[startFrame],
                bassIndex = bassPath[startFrame],
            )
            startFrame = frame
            lastKey = key
        }
        raw += AnimeChordEvent(
            startSeconds = startFrame * secondsPerFrame,
            // Upstream's final event ends one frame past the last frame's timestamp.
            endSeconds = (frames - 1) * secondsPerFrame + secondsPerFrame,
            rootChordIndex = rootPath[startFrame],
            bassIndex = bassPath[startFrame],
        )

        if (minChordSeconds <= 0.0) return raw

        val filtered = ArrayList<AnimeChordEvent>(raw.size)
        for (event in raw) {
            val duration = event.endSeconds - event.startSeconds
            if (duration < minChordSeconds && filtered.isNotEmpty()) {
                val last = filtered.size - 1
                filtered[last] = filtered[last].copy(endSeconds = event.endSeconds)
                if (filtered.size >= 2 && sameChord(filtered[last - 1], filtered[last], vocabulary)) {
                    filtered[last - 1] = filtered[last - 1].copy(endSeconds = filtered[last].endSeconds)
                    filtered.removeAt(last)
                }
            } else {
                val last = filtered.lastOrNull()
                if (last != null && sameChord(last, event, vocabulary)) {
                    filtered[filtered.size - 1] = last.copy(endSeconds = event.endSeconds)
                } else {
                    filtered += event
                }
            }
        }
        return filtered
    }

    private fun sameChord(
        first: AnimeChordEvent,
        second: AnimeChordEvent,
        vocabulary: AnimeChordVocabulary,
    ): Boolean = vocabulary.chordKey(first.rootChordIndex, first.bassIndex) ==
        vocabulary.chordKey(second.rootChordIndex, second.bassIndex)
}
