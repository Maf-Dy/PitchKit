package com.nicos.pitchkit.tuner.harmony.animechord

import org.json.JSONObject

internal class AnimeChordQuality(
    val index: Int,
    /** The dlchordx spelling the model's own vocabulary uses, e.g. `m7(9)`. */
    val native: String,
    /** Semitones above the root, reduced mod 12 and sorted; always starts at 0. */
    val intervals: IntArray,
    /** Harte shorthand, e.g. `min9`; see [AnimeChordVocabulary]. */
    val shorthand: String,
)

/**
 * The model's 745-class root-chord vocabulary, baked from the checkpoint's own
 * `label_vocab` by `tools/generate-animechord-cqt-plans.py`.
 *
 * The model emits `<root><quality>[/<bass>]` in **dlchordx** spelling over 62 qualities.
 * A Kotlin port cannot call dlchordx, so each quality ships as the interval set dlchordx
 * parses it to, together with the Harte shorthand
 * `tools/accuracy_audit/live_metrics.canonical_harte` derives from that interval set —
 * the same canonicalisation the review server applies to every anime lane before it
 * scores or draws one. That is why a chord reads `C:min9` here and `C:min9` there, and
 * why `Eb:maj7/5` carries its bass as a Harte degree rather than a note name.
 *
 * Index arithmetic is `PredictionDecoder._build_root_chord_labels`: roots 1..12 outer,
 * the 62 non-`N` qualities inner, then one final `N` class.
 */
internal class AnimeChordVocabulary(
    val qualities: List<AnimeChordQuality>,
    /** 13 entries, index 0 = `N`, then C..B. */
    val pitchClassLabels: List<String>,
    /** 12 Harte degrees for a bass a given number of semitones above the root. */
    val bassDegrees: List<String>,
    val rootChordCount: Int,
) {
    val noChordIndex: Int = rootChordCount - 1

    init {
        require(qualities.size == AnimeChordContract.QUALITY_COUNT) {
            "anime vocabulary must hold ${AnimeChordContract.QUALITY_COUNT} qualities"
        }
        require(pitchClassLabels.size == 13 && pitchClassLabels[0] == "N")
        require(bassDegrees.size == 12)
        require(rootChordCount == AnimeChordContract.ROOT_COUNT * qualities.size + 1)
        qualities.forEachIndexed { index, quality ->
            require(quality.index == index)
            require(quality.intervals.isNotEmpty() && quality.intervals[0] == 0)
            require(quality.shorthand.isNotEmpty())
        }
    }

    fun rootPitchClass(rootChordIndex: Int): Int? =
        if (rootChordIndex == noChordIndex) null else rootChordIndex / qualities.size

    fun quality(rootChordIndex: Int): AnimeChordQuality? =
        if (rootChordIndex == noChordIndex) null else qualities[rootChordIndex % qualities.size]

    /**
     * The bass pitch class this frame actually shows, or null when the bass head says
     * `N` or repeats the root — `PredictionDecoder.decode_frames` drops the slash in both
     * cases, so the two collapse to the same chord.
     */
    fun bassPitchClass(rootChordIndex: Int, bassIndex: Int): Int? {
        val root = rootPitchClass(rootChordIndex) ?: return null
        if (bassIndex <= 0 || bassIndex >= pitchClassLabels.size) return null
        val bass = bassIndex - 1
        return if (bass == root) null else bass
    }

    /**
     * A key that is equal exactly when `decode_frames` would produce the same chord
     * string, so runs can be found without building a string per frame.
     */
    fun chordKey(rootChordIndex: Int, bassIndex: Int): Int {
        if (rootChordIndex == noChordIndex) return -1
        val bass = bassPitchClass(rootChordIndex, bassIndex)
        return rootChordIndex * 13 + 1 + (bass ?: -1)
    }

    /** The model's own spelling, e.g. `Ebm7(9)/G` — what the audit calls `raw_label`. */
    fun nativeLabel(rootChordIndex: Int, bassIndex: Int): String {
        val root = rootPitchClass(rootChordIndex) ?: return "N"
        val quality = qualities[rootChordIndex % qualities.size]
        val text = pitchClassLabels[root + 1] + quality.native
        val bass = bassPitchClass(rootChordIndex, bassIndex) ?: return text
        return "$text/${pitchClassLabels[bass + 1]}"
    }

    /** Canonical Harte, e.g. `Eb:min9/3`; null for the no-chord class. */
    fun harteLabel(rootChordIndex: Int, bassIndex: Int): String? {
        val root = rootPitchClass(rootChordIndex) ?: return null
        val quality = qualities[rootChordIndex % qualities.size]
        // The app's chord parser (ChordTheory.parseChord) reads Harte shorthand with a
        // NOTE after the slash and interval lists in parentheses, but not a degree bass
        // (`/b7`) and not the three bare power-chord spellings. Spell those the way the
        // parser reads them, so the Songs screen and the instrument diagram can use every
        // label this engine emits. The interval set is unchanged; only the spelling is.
        val shorthand = when (quality.shorthand) {
            "5" -> "(1,5)"
            "1(3,b5)" -> "(1,3,b5)"
            "1(3,b5,b7)" -> "(1,3,b5,b7)"
            else -> quality.shorthand
        }
        val text = pitchClassLabels[root + 1] + ":" + shorthand
        val bass = bassPitchClass(rootChordIndex, bassIndex) ?: return text
        return text + "/" + pitchClassLabels[bass + 1]
    }

    /** Sounding pitch classes, in rising semitone order from the root. */
    fun pitchClassNames(rootChordIndex: Int, bassIndex: Int): List<String> {
        val root = rootPitchClass(rootChordIndex) ?: return emptyList()
        val quality = qualities[rootChordIndex % qualities.size]
        val names = quality.intervals.map { pitchClassLabels[(root + it) % 12 + 1] }
        val bass = bassPitchClass(rootChordIndex, bassIndex) ?: return names
        val bassName = pitchClassLabels[bass + 1]
        return if (bassName in names) names else listOf(bassName) + names
    }

    fun noteName(pitchClass: Int): String = pitchClassLabels[(pitchClass % 12) + 1]

    companion object {
        fun parse(json: String): AnimeChordVocabulary {
            val root = JSONObject(json)
            require(root.getInt("schema_version") == 1)
            require(root.getString("checkpoint_sha256") == AnimeChordContract.CHECKPOINT_SHA256) {
                "anime vocabulary was baked from a different checkpoint"
            }
            val labels = root.getJSONArray("pitch_class_labels")
            val degrees = root.getJSONArray("bass_degrees")
            val array = root.getJSONArray("qualities")
            return AnimeChordVocabulary(
                qualities = List(array.length()) { index ->
                    val item = array.getJSONObject(index)
                    val intervals = item.getJSONArray("intervals")
                    AnimeChordQuality(
                        index = item.getInt("index"),
                        native = item.getString("native"),
                        intervals = IntArray(intervals.length()) { intervals.getInt(it) },
                        shorthand = item.getString("shorthand"),
                    )
                },
                pitchClassLabels = List(labels.length()) { labels.getString(it) },
                bassDegrees = List(degrees.length()) { degrees.getString(it) },
                rootChordCount = root.getInt("root_classes"),
            )
        }
    }
}
