package com.nicos.pitchkit.tuner.harmony.lvchordia

internal class LvChordiaLabelFormatter(
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

        val normalized = when {
            suffix == "maj" -> ""
            suffix == "min" -> "m"
            suffix == "maj6" -> "6"
            suffix == "min6" -> "m6"
            suffix == "maj6(9)" || suffix == "maj(6,9)" -> "6/9"
            suffix == "min6(9)" || suffix == "min(6,9)" -> "m6/9"
            suffix == "hdim7" -> "ø7"
            suffix == "min7" -> "m7"
            suffix == "min9" -> "m9"
            suffix == "min11" -> "m11"
            suffix == "min13" -> "m13"
            suffix == "minmaj7" -> "m(maj7)"
            suffix.startsWith("maj(") -> suffix.removePrefix("maj")
            suffix.startsWith("min(") -> "m" + suffix.removePrefix("min")
            else -> suffix
        }

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
