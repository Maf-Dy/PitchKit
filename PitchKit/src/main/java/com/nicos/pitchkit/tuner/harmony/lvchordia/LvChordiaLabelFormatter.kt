package com.nicos.pitchkit.tuner.harmony.lvchordia

class LvChordiaLabelFormatter(
    private val preferFlats: Boolean = false,
) {
    private val sharps = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    private val flats = arrayOf("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
    private val rootMap = mapOf(
        "C" to 0, "B#" to 0, "C#" to 1, "Db" to 1, "D" to 2,
        "D#" to 3, "Eb" to 3, "E" to 4, "Fb" to 4, "E#" to 5,
        "F" to 5, "F#" to 6, "Gb" to 6, "G" to 7, "G#" to 8,
        "Ab" to 8, "A" to 9, "A#" to 10, "Bb" to 10, "B" to 11, "Cb" to 11,
    )

    private val qualityNames = mapOf(
        "maj" to "",
        "min" to "m",
        "maj6" to "6",
        "min6" to "m6",
        "maj6(9)" to "6/9",
        "maj(6,9)" to "6/9",
        "min6(9)" to "m6/9",
        "min(6,9)" to "m6/9",
        "hdim7" to "ø7",
        "min7" to "m7",
        "min9" to "m9",
        "min11" to "m11",
        "min13" to "m13",
        "minmaj7" to "m(maj7)",
        "sus4(b7)" to "7sus4",
        "sus4(b7,9)" to "9sus4",
        "sus4(b7,9,13)" to "13sus4",
    )

    fun format(raw: String): String {
        if (raw == "N" || raw == "X") return raw
        val separator = raw.indexOf(':')
        if (separator <= 0) return raw
        val rootToken = raw.substring(0, separator)
        val rootPc = rootMap[rootToken] ?: return raw
        var suffix = raw.substring(separator + 1)

        var slash: String? = null
        val slashIndex = suffix.lastIndexOf('/')
        if (slashIndex >= 0) {
            slash = suffix.substring(slashIndex + 1)
            suffix = suffix.substring(0, slashIndex)
        }

        val normalized = normalizeQuality(suffix)

        return buildString {
            append(noteName(rootPc))
            append(normalized)
            val bassInterval = slash?.let(::degreeToSemitone)
            if (bassInterval != null) {
                val bassPc = (rootPc + bassInterval) % 12
                if (bassPc != rootPc) {
                    append('/')
                    append(noteName(bassPc))
                }
            }
        }
    }

    /**
     * Harte quality -> display quality. Suspensions are never allowed to borrow a third:
     * sus4(b7) is a 7sus4, not a m7. Any quality carrying an extension list that has no
     * dedicated spelling keeps that list verbatim after the mapped base quality, so the
     * label stays a truthful record of the dictionary entry.
     */
    private fun normalizeQuality(suffix: String): String {
        qualityNames[suffix]?.let { return it }
        val open = suffix.indexOf('(')
        if (open >= 0 && suffix.endsWith(')')) {
            val base = suffix.substring(0, open)
            return (qualityNames[base] ?: base) + suffix.substring(open)
        }
        return suffix
    }

    private fun noteName(pc: Int): String = if (preferFlats) flats[pc] else sharps[pc]

    private fun degreeToSemitone(token: String): Int? {
        if (token.isBlank()) return null
        var index = 0
        var accidental = 0
        while (index < token.length && (token[index] == 'b' || token[index] == '#')) {
            accidental += if (token[index] == 'b') -1 else 1
            index++
        }
        val degree = token.substring(index).toIntOrNull() ?: return null
        val zeroBased = degree - 1
        if (zeroBased < 0) return null
        val octaves = zeroBased / 7
        val scaleIndex = zeroBased % 7
        val majorScale = intArrayOf(0, 2, 4, 5, 7, 9, 11)
        return ((majorScale[scaleIndex] + 12 * octaves + accidental) % 12 + 12) % 12
    }
}
