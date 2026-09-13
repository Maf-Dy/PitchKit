package com.nicos.pitchkit.tuner.harmony.chordnet

/** Exact 170-label ordering used by the ChordMini runner. */
object ChordNetVocabulary {
    private val roots = listOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    private val suffixes = listOf(
        "min", "", "dim", "aug", "min6", "maj6", "min7", "minmaj7",
        "maj7", "7", "dim7", "hdim7", "sus2", "sus4",
    )

    val labels: List<String> = buildList {
        roots.forEach { root ->
            suffixes.forEach { suffix ->
                add(if (suffix.isEmpty()) root else "$root:$suffix")
            }
        }
        add("X")
        add("N")
    }

    init {
        check(labels.size == ChordNetContract.CHORD_COUNT)
    }

    fun labelAt(index: Int): String = labels.getOrElse(index) { "N" }

    fun displayLabel(index: Int): String? = when (val label = labelAt(index)) {
        "N", "X" -> null
        else -> toDisplay(label)
    }

    fun toDisplay(label: String): String {
        val parts = label.split(':', limit = 2)
        if (parts.size == 1) return parts[0]
        val root = parts[0]
        return root + when (parts[1]) {
            "min" -> "m"
            "dim" -> "dim"
            "aug" -> "aug"
            "min6" -> "m6"
            "maj6" -> "6"
            "min7" -> "m7"
            "minmaj7" -> "m(maj7)"
            "maj7" -> "maj7"
            "7" -> "7"
            "dim7" -> "dim7"
            "hdim7" -> "ø7"
            "sus2" -> "sus2"
            "sus4" -> "sus4"
            else -> ":${parts[1]}"
        }
    }
}
