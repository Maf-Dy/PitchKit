package com.nicos.pitchkit.tuner.harmony.solitito

/**
 * solitito's nine chord qualities plus two meta classes, and the display labels they
 * become.
 *
 * The model has a 13-way root head (twelve pitch classes plus `Noise`) and an 11-way
 * quality head. Nine of those eleven are chords; `note` means *"that is a single note,
 * not a chord"* and `N` means nothing at all. Both, and the `Noise` root, put nothing on
 * screen, which the app writes as `N`.
 *
 * `dim7` is the trap. `brain.rs` spells the fully diminished seventh `dim`, and reading
 * that as a diminished **triad** would silently drop the diminished seventh from every
 * extension metric — the label must be `dim7` (Harte `dim7`, `(0,3,6,9)`), and `m7b5` is
 * Harte's `hdim7`, `(0,3,6,10)`. The two are the app's `m7b5` / `dim7` display suffixes,
 * which `ChordPracticeLevel` and the live screen already parse.
 *
 * There are no 6th chords, no ninths, elevenths or thirteenths, no inversions and no bass
 * head in this vocabulary at all. That is a ceiling, and it is also why solitito's
 * extension evidence never over-claims: it cannot invent a tension it did not hear.
 */
internal object SolititoVocabulary {
    /** Head index 12 is `Noise`; sharps only, because the model's classes are. */
    val ROOTS = arrayOf(
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "Noise",
    )

    val QUALITIES = arrayOf(
        "maj", "min", "maj7", "dom7", "min7", "m7b5", "dim7", "aug", "sus", "note", "N",
    )

    /** `brain.rs::quality_suffix` — the internal identity the vote and the latch use. */
    private val INTERNAL_SUFFIX = mapOf(
        "maj" to "", "min" to "m", "maj7" to "Maj7", "dom7" to "7", "min7" to "m7",
        "m7b5" to "m7b5", "dim7" to "dim", "aug" to "aug", "sus" to "sus4",
    )

    /** ...and the same nine onto the suffixes the app's own label parser reads. */
    private val DISPLAY_SUFFIX = mapOf(
        "" to "", "m" to "m", "Maj7" to "maj7", "7" to "7", "m7" to "m7",
        "m7b5" to "m7b5", "dim" to "dim7", "aug" to "aug", "sus4" to "sus4",
    )

    /** `latch.rs`: the qualities that ring on as a four-note chord. */
    private val FOUR_NOTE = setOf("Maj7", "7", "m7", "m7b5", "dim")

    /** Nothing on screen. */
    const val NO_CHORD = "N"

    /** The internal marker for "asked, but the window is not live enough to name". */
    const val PENDING = "..."

    const val NOISE = "Noise"

    /**
     * The internal chord identity, exactly as `brain.rs` builds the string: `"Noise"`,
     * `"Note C"`, or `"C"` / `"C m7"`. Kept as a string rather than a pair because the
     * vote, the latch and the decay test all key off it and a transcription that shares
     * the reference's identity cannot drift from it.
     */
    fun chordString(rootIndex: Int, quality: String): String {
        if (rootIndex >= 12 || rootIndex < 0 || quality == "N") return NOISE
        val root = ROOTS[rootIndex]
        if (quality == "note") return "Note $root"
        val suffix = INTERNAL_SUFFIX[quality] ?: return NOISE
        return if (suffix.isEmpty()) root else "$root $suffix"
    }

    fun rootOf(chord: String): String {
        val parts = chord.split(' ')
        if (parts.isEmpty()) return ""
        return if (parts[0] == "Note") parts.getOrElse(1) { "" } else parts[0]
    }

    fun qualityOf(chord: String): String {
        val parts = chord.split(' ')
        if (parts.isEmpty() || parts[0] == "Note") return ""
        return parts.getOrElse(1) { "" }
    }

    /** A chord the latch may lock onto: not noise, not pending, not a bare note. */
    fun isRealChord(chord: String): Boolean =
        chord.isNotEmpty() && chord != NOISE && chord != PENDING && !chord.startsWith("Note")

    /**
     * Whether [now] is the decaying tail of [held]: the same root, and a four-note chord
     * that has lost a note. A `Note C` reading counts, which is deliberate — it is what
     * a `C m7` sounds like once three of its four notes have died away.
     */
    fun isDecayOf(held: String, now: String): Boolean =
        rootOf(held) == rootOf(now) &&
            qualityOf(held) in FOUR_NOTE &&
            qualityOf(now) !in FOUR_NOTE

    /**
     * The internal identity as the live screen renders it: `N` for anything that is not a
     * named chord, otherwise root plus the app's own suffix.
     */
    fun toDisplayLabel(chord: String): String {
        if (chord.isEmpty() || chord == NOISE || chord == PENDING || chord.startsWith("Note")) {
            return NO_CHORD
        }
        val suffix = DISPLAY_SUFFIX[qualityOf(chord)]
            ?: error("Unmapped solitito prediction '$chord'")
        return rootOf(chord) + suffix
    }
}
