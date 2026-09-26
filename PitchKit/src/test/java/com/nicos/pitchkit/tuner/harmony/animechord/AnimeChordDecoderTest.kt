package com.nicos.pitchkit.tuner.harmony.animechord

import kotlin.math.exp
import kotlin.math.ln
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The decode half: the sticky Viterbi against a dense textbook one, and the label and
 * event builders against a whole real song from the PC lane.
 */
class AnimeChordDecoderTest {
    private val vocabulary = AnimeChordVocabulary.parse(
        AnimeChordTestAssets.read(AnimeChordContract.VOCABULARY_FILE).toString(Charsets.UTF_8)
    )

    /**
     * `anime-windowed-report.md` §8.2's claim is that a uniform off-diagonal collapses the
     * author's O(T·C²) Viterbi to O(T·C) with the same path. That is the whole reason this
     * engine can decode 745 states on a phone, so it is checked against a dense Viterbi
     * that makes no such assumption — including at `stay_prob = 0.9`, where the switch
     * penalty is small enough that the path really does jump about.
     */
    @Test
    fun theStickyViterbiAgreesWithADenseOne() {
        for ((classes, stay) in listOf(2 to 1.0, 7 to 1.0, 40 to 1.0, 40 to 0.9, 13 to 0.5)) {
            val frames = 240
            val logits = pseudoRandomLogits(frames, classes, seed = 0x9E3779B97F4A7C15uL)
            val logStay = ln(stay + 1e-9)
            val logSwitch = ln((1.0 - stay) / (classes - 1) + 1e-9)

            val fast = AnimeChordViterbi(classes, logStay, logSwitch, frames)
            for (frame in 0 until frames) fast.add(logits, frame * classes)

            assertArrayEquals(
                "classes=$classes stay=$stay",
                denseViterbi(logits, frames, classes, logStay, logSwitch),
                fast.decode(),
            )
        }
    }

    /**
     * The emission is `log(softmax + 1e-9)` with the softmax **rounded to float32 first**.
     * With 745 competing classes a confident frame pushes most of them below float32's
     * smallest subnormal, and upstream's float32 softmax turns those into exactly zero, so
     * their emission is the `ln(1e-9)` floor rather than the -80 or so a double softmax
     * would produce. Getting that wrong changes which state survives a long stay.
     */
    @Test
    fun anUnderflowedClassSitsOnTheEpsilonFloor() {
        val classes = 745
        val logits = FloatArray(classes)
        logits[17] = 400f
        val viterbi = AnimeChordViterbi(classes, expectedFrames = 2)
        viterbi.add(logits)
        viterbi.add(logits)
        assertArrayEquals("saturated", intArrayOf(17, 17), viterbi.decode())

        // A second frame that prefers another class by less than the switch penalty must
        // not move the path; by more than it, it must.
        val floor = ln(1e-9)
        assertEquals(AnimeChordContract.LOG_SWITCH, floor, 0.0)
        assertTrue(floor < -20.0 && floor > -21.0)
    }

    @Test
    fun aWholeSongDecodesToTheSegmentsThePcLanePublished() {
        val fixture = JSONObject(
            resource("animechord-decode-fixture.json").toString(Charsets.UTF_8)
        )
        assertEquals(1, fixture.getInt("schema_version"))
        assertEquals("anime-hmm-w30-o50", fixture.getString("lane"))
        val frames = fixture.getInt("frames")
        val secondsPerFrame = fixture.getDouble("seconds_per_frame")
        assertEquals(AnimeChordContract.SECONDS_PER_FRAME, secondsPerFrame, 1e-12)
        assertEquals(
            AnimeChordContract.MIN_CHORD_SECONDS,
            fixture.getDouble("min_duration_chord"),
            0.0,
        )

        val rootPath = expand(fixture.getJSONArray("root_runs"), frames)
        val bassPath = expand(fixture.getJSONArray("bass_runs"), frames)
        val events = AnimeChordDecoder.events(
            rootPath = rootPath,
            bassPath = bassPath,
            vocabulary = vocabulary,
            secondsPerFrame = secondsPerFrame,
        )

        val expected = fixture.getJSONArray("expected_segments")
        assertEquals("segment count", expected.length(), events.size)
        for (index in 0 until expected.length()) {
            val want = expected.getJSONObject(index)
            val got = events[index]
            assertEquals(
                "segment $index start",
                want.getDouble("start"),
                got.startSeconds,
                5e-4,
            )
            assertEquals("segment $index end", want.getDouble("end"), got.endSeconds, 5e-4)
            // The PC lane spells the bass as a degree (`E:min/4`); the port spells it as
            // the note the app parser reads (`E:min/A`). Same chord, one spelling rule.
            assertEquals(
                "segment $index label",
                noteBass(want.getString("label")),
                vocabulary.harteLabel(got.rootChordIndex, got.bassIndex) ?: "N",
            )
            assertEquals(
                "segment $index raw label",
                want.getString("raw_label"),
                vocabulary.nativeLabel(got.rootChordIndex, got.bassIndex),
            )
        }
        // The song really does use the vocabulary this engine exists for.
        assertTrue(
            "no extended chord in the fixture",
            (0 until expected.length()).any {
                val label = expected.getJSONObject(it).getString("label")
                label.contains("9") || label.contains("11") || label.contains("13")
            },
        )
    }

    @Test
    fun aRunShorterThanATenthOfASecondIsFoldedIntoItsPredecessor() {
        val cMaj = vocabulary.qualities.first { it.native == "" }.index
        val gMaj = 7 * 62 + cMaj
        val frames = IntArray(20) { if (it in 10..11) gMaj else cMaj }
        // 2 frames is 0.046 s, under the 0.1 s floor, so it is absorbed and the two C
        // runs either side become one.
        val folded = AnimeChordDecoder.events(frames, IntArray(20), vocabulary)
        assertEquals(1, folded.size)
        assertEquals(cMaj, folded.single().rootChordIndex)

        // Six frames is 0.139 s and survives as its own event -- as long as what comes
        // after it is also long enough, because upstream folds a short *tail* forwards
        // into its predecessor too.
        val kept = AnimeChordDecoder.events(
            IntArray(30) { if (it in 10..15) gMaj else cMaj },
            IntArray(30),
            vocabulary,
        )
        assertEquals(3, kept.size)
        assertEquals(listOf(cMaj, gMaj, cMaj), kept.map { it.rootChordIndex })
        assertEquals(
            30 * AnimeChordContract.SECONDS_PER_FRAME,
            kept.last().endSeconds,
            1e-9,
        )

        // A four-frame tail (0.093 s) is absorbed by the chord before it, which keeps
        // that chord's label -- the asymmetry upstream's filter really has.
        val tail = AnimeChordDecoder.events(
            IntArray(20) { if (it in 10..15) gMaj else cMaj },
            IntArray(20),
            vocabulary,
        )
        assertEquals(listOf(cMaj, gMaj), tail.map { it.rootChordIndex })
        assertEquals(
            20 * AnimeChordContract.SECONDS_PER_FRAME,
            tail.last().endSeconds,
            1e-9,
        )
    }

    // ------------------------------------------------------------------ helpers

    private fun expand(runs: JSONArray, frames: Int): IntArray {
        val output = IntArray(frames)
        var position = 0
        for (index in 0 until runs.length()) {
            val run = runs.getJSONArray(index)
            repeat(run.getInt(1)) { output[position++] = run.getInt(0) }
        }
        assertEquals("run-length total", frames, position)
        return output
    }

    /** A textbook O(T·C²) Viterbi over the full transition matrix. */
    private fun denseViterbi(
        logits: FloatArray,
        frames: Int,
        classes: Int,
        logStay: Double,
        logSwitch: Double,
    ): IntArray {
        val emission = Array(frames) { frame -> logSoftmax(logits, frame * classes, classes) }
        // float32 scores, as the reference's numba kernel declares them, so the two
        // implementations round identically and the comparison is about the recurrence.
        val score = Array(frames) { FloatArray(classes) }
        val back = Array(frames) { IntArray(classes) }
        val initial = ln(1.0 / classes + 1e-9)
        for (state in 0 until classes) {
            score[0][state] = (initial + emission[0][state]).toFloat()
        }
        for (frame in 1 until frames) {
            for (state in 0 until classes) {
                var best = Double.NEGATIVE_INFINITY
                var bestFrom = -1
                for (from in 0 until classes) {
                    val value = score[frame - 1][from].toDouble() +
                        (if (from == state) logStay else logSwitch)
                    if (value > best) {
                        best = value
                        bestFrom = from
                    }
                }
                score[frame][state] = (best + emission[frame][state]).toFloat()
                back[frame][state] = bestFrom
            }
        }
        val path = IntArray(frames)
        var last = 0
        for (state in 1 until classes) if (score[frames - 1][state] > score[frames - 1][last]) last = state
        path[frames - 1] = last
        for (frame in frames - 1 downTo 1) path[frame - 1] = back[frame][path[frame]]
        return path
    }

    private fun logSoftmax(logits: FloatArray, offset: Int, classes: Int): DoubleArray {
        var maximum = logits[offset]
        for (index in 1 until classes) {
            if (logits[offset + index] > maximum) maximum = logits[offset + index]
        }
        val values = DoubleArray(classes)
        var total = 0.0
        for (index in 0 until classes) {
            values[index] = exp((logits[offset + index] - maximum).toDouble())
            total += values[index]
        }
        for (index in 0 until classes) {
            values[index] = ln((values[index] / total).toFloat().toDouble() + 1e-9)
        }
        return values
    }

    /**
     * A deterministic generator, so the comparison covers hundreds of frames without a
     * fixture. `splitmix64` is arithmetic, not a library, so it produces the same stream
     * everywhere.
     */
    private fun pseudoRandomLogits(frames: Int, classes: Int, seed: ULong): FloatArray {
        var state = seed
        return FloatArray(frames * classes) {
            state += 0x9E3779B97F4A7C15uL
            var z = state
            z = (z xor (z shr 30)) * 0xBF58476D1CE4E5B9uL
            z = (z xor (z shr 27)) * 0x94D049BB133111EBuL
            z = z xor (z shr 31)
            (((z shr 40).toDouble() / (1 shl 24).toDouble()) * 16.0 - 8.0).toFloat()
        }
    }

    private fun assertArrayEquals(message: String, expected: IntArray, actual: IntArray) {
        assertEquals("$message length", expected.size, actual.size)
        var mismatches = 0
        var first = -1
        for (index in expected.indices) {
            if (expected[index] != actual[index]) {
                if (first < 0) first = index
                mismatches++
            }
        }
        assertEquals("$message: $mismatches differ, first at $first", 0, mismatches)
    }

    private fun resource(name: String) =
        requireNotNull(javaClass.getResourceAsStream("/$name")) { name }.use { it.readBytes() }

    private companion object {
        private val NOTE_INDEX = mapOf(
            "C" to 0, "Db" to 1, "C#" to 1, "D" to 2, "Eb" to 3, "D#" to 3, "E" to 4, "F" to 5,
            "Gb" to 6, "F#" to 6, "G" to 7, "Ab" to 8, "G#" to 8, "A" to 9, "Bb" to 10,
            "A#" to 10, "B" to 11,
        )
        private val DEGREE_SEMITONES = listOf("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")
        private val FLAT_NAMES = listOf("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

        /** `E:min/4` -> `E:min/A`; labels without a degree bass are returned unchanged. */
        fun noteBass(label: String): String {
            val slash = label.lastIndexOf('/')
            if (slash < 0) return label
            val degree = label.substring(slash + 1)
            val semis = DEGREE_SEMITONES.indexOf(degree)
            if (semis < 0) return label
            val root = NOTE_INDEX[label.substringBefore(':')] ?: return label
            return label.substring(0, slash + 1) + FLAT_NAMES[(root + semis) % 12]
        }
    }
}
