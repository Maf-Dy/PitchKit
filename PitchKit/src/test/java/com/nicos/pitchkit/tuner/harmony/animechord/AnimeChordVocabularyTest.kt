package com.nicos.pitchkit.tuner.harmony.animechord

import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The shipped assets and the 745-class label arithmetic.
 *
 * The vocabulary is the one piece of this port that had to be re-expressed rather than
 * carried over: the model spells chords with `dlchordx`, which does not exist on a phone,
 * so `tools/generate-animechord-cqt-plans.py` bakes each quality's interval set and the
 * Harte shorthand `live_metrics.canonical_harte` derives from it. This pins the index
 * arithmetic, the bass rule and a sample of the spellings the review server produces.
 */
class AnimeChordVocabularyTest {
    private val vocabulary = AnimeChordVocabulary.parse(
        AnimeChordTestAssets.read(AnimeChordContract.VOCABULARY_FILE).toString(Charsets.UTF_8)
    )

    @Test
    fun theMetadataAgreesWithTheCompiledContract() {
        val meta = JSONObject(
            AnimeChordTestAssets.read(AnimeChordContract.METADATA_FILE).toString(Charsets.UTF_8)
        )
        assertEquals(1, meta.getInt("schema_version"))
        assertEquals(AnimeChordContract.BACKEND, meta.getString("backend"))
        assertEquals(AnimeChordContract.CHECKPOINT_SHA256, meta.getString("checkpoint_sha256"))
        assertEquals(AnimeChordContract.HF_REVISION, meta.getString("hf_revision"))
        assertEquals(AnimeChordContract.SOURCE_COMMIT, meta.getString("source_commit"))
        assertEquals(AnimeChordContract.MODEL_SHA256, meta.getString("model_sha256"))
        assertEquals(AnimeChordContract.WINDOW_FRAMES, meta.getInt("window_frames"))
        assertEquals(AnimeChordContract.OVERLAP_FRAMES, meta.getInt("overlap_frames"))
        assertTrue(meta.getBoolean("dynamic_time_axis"))
        assertTrue(!meta.getBoolean("first_window_primed"))
        assertEquals(
            AnimeChordContract.CQT_PLAN_SHA256,
            meta.getJSONObject("cqt").getString("sha256"),
        )
        assertEquals(
            AnimeChordContract.VOCABULARY_SHA256,
            meta.getJSONObject("vocabulary").getString("sha256"),
        )
        assertEquals(
            AnimeChordContract.MODEL_SHA256,
            AnimeChordTestAssets.sha256(AnimeChordContract.MODEL_FILE),
        )
        assertEquals(
            AnimeChordContract.CQT_PLAN_SHA256,
            AnimeChordTestAssets.sha256(AnimeChordContract.CQT_PLAN_FILE),
        )
        assertEquals(
            AnimeChordContract.VOCABULARY_SHA256,
            AnimeChordTestAssets.sha256(AnimeChordContract.VOCABULARY_FILE),
        )
    }

    @Test
    fun theCqtPlanIsTheOneTheContractDescribes() {
        val plan = AnimeChordCqtPlanDecoder.decodeAndVerify(
            AnimeChordTestAssets.read(AnimeChordContract.CQT_PLAN_FILE)
        )
        assertEquals(AnimeChordContract.SAMPLE_RATE.toDouble(), plan.sampleRate, 0.0)
        assertEquals(AnimeChordContract.HOP_LENGTH, plan.hopLength)
        assertEquals(AnimeChordContract.INPUT_BINS, plan.binCount)
        assertEquals(AnimeChordContract.OCTAVES, plan.octaveCount)
        assertEquals(AnimeChordContract.STAGE_FFT, plan.stageFftSize)
        assertEquals(AnimeChordContract.STAGE_BINS, plan.stageBinCount)
        assertEquals(AnimeChordContract.DECIMATOR_TAPS, plan.decimator.size)
        assertEquals(AnimeChordContract.FMIN_HZ, plan.fmin, 1e-9)
        // Q = filter_scale / (2^(1/36) - 1).
        assertEquals(
            AnimeChordContract.FILTER_SCALE / (Math.pow(2.0, 1.0 / 36.0) - 1.0),
            plan.q,
            1e-9,
        )
        // firwin normalises the low-pass to unit DC gain.
        assertEquals(1.0, plan.decimator.sumOf { it.toDouble() }, 1e-6)
        // A periodic hann window starts at zero and peaks at one in the middle.
        assertEquals(0.0f, plan.window[0], 0f)
        assertEquals(1.0f, plan.window[AnimeChordContract.STAGE_FFT / 2], 1e-6f)
        assertEquals(
            AnimeChordContract.OCTAVES * AnimeChordContract.BINS_PER_OCTAVE *
                AnimeChordContract.STAGE_BINS * 2,
            plan.kernels.size,
        )
    }

    @Test
    fun theRootChordIndexArithmeticIsUpstreams() {
        assertEquals(AnimeChordContract.ROOT_CHORD_COUNT, vocabulary.rootChordCount)
        assertEquals(744, vocabulary.noChordIndex)
        assertEquals(62, vocabulary.qualities.size)

        // Index 0 is C with the first quality slot, `5`; the roots move every 62 slots.
        assertEquals(0, vocabulary.rootPitchClass(0))
        assertEquals("5", vocabulary.quality(0)?.native)
        assertEquals(1, vocabulary.rootPitchClass(62))
        assertEquals("5", vocabulary.quality(62)?.native)
        assertEquals(11, vocabulary.rootPitchClass(743))
        assertNull(vocabulary.rootPitchClass(744))
        assertNull(vocabulary.harteLabel(744, 0))
    }

    @Test
    fun aChordReadsTheWayTheReviewServerSpellsIt() {
        fun index(root: Int, native: String): Int =
            root * 62 + vocabulary.qualities.first { it.native == native }.index

        // `C:min9` -- the label the task statement asks for, and what
        // live_metrics.canonical_harte makes of `C:(1,b3,5,b7,9)`.
        assertEquals("C:min9", vocabulary.harteLabel(index(0, "m7(9)"), 0))
        // `Eb:maj7/5`: bass a fifth above the root, written as a Harte degree.
        assertEquals("Eb:maj7/Bb", vocabulary.harteLabel(index(3, "M7"), 11))
        assertEquals("Ebm7(9)", vocabulary.nativeLabel(index(3, "m7(9)"), 0))
        assertEquals("Ebm7(9)/Gb", vocabulary.nativeLabel(index(3, "m7(9)"), 7))
        // The plain qualities keep their plain shorthand.
        assertEquals("G:maj", vocabulary.harteLabel(index(7, ""), 0))
        assertEquals("A:min7", vocabulary.harteLabel(index(9, "m7"), 0))
        assertEquals("B:7", vocabulary.harteLabel(index(11, "7"), 0))
        assertEquals("F:13", vocabulary.harteLabel(index(5, "7(9,13)"), 0))
        assertEquals("D:hdim7", vocabulary.harteLabel(index(2, "m7-5"), 0))
    }

    @Test
    fun aBassThatRepeatsTheRootOrSaysNoChordDropsTheSlash() {
        val cMin9 = vocabulary.qualities.first { it.native == "m7(9)" }.index
        // bass index 0 is the `N` class, index 1 is C.
        assertEquals("C:min9", vocabulary.harteLabel(cMin9, 0))
        assertEquals("C:min9", vocabulary.harteLabel(cMin9, 1))
        assertEquals("C:min9/Eb", vocabulary.harteLabel(cMin9, 4))
        assertNull(vocabulary.bassPitchClass(cMin9, 1))
        assertEquals(3, vocabulary.bassPitchClass(cMin9, 4))
        // Both no-slash spellings have to collapse to one run key, or the event builder
        // would split a chord in two where upstream does not.
        assertEquals(vocabulary.chordKey(cMin9, 0), vocabulary.chordKey(cMin9, 1))
        assertTrue(vocabulary.chordKey(cMin9, 0) != vocabulary.chordKey(cMin9, 4))
        assertEquals(-1, vocabulary.chordKey(744, 7))
    }

    @Test
    fun everyQualityHasADistinctIntervalSetAndAParsableShorthand() {
        val seen = mutableMapOf<Set<Int>, String>()
        for (quality in vocabulary.qualities) {
            val intervals = quality.intervals.toSet()
            assertTrue("duplicate interval set for ${quality.native}", intervals !in seen)
            seen[intervals] = quality.native
            assertTrue(quality.intervals.all { it in 0..11 })
            assertEquals(quality.intervals.size, intervals.size)
            assertTrue(quality.shorthand.none { it.isWhitespace() })
        }
        assertEquals(62, seen.size)
        // The pitch classes a segment reports are the quality's intervals over its root.
        val ninth = vocabulary.qualities.first { it.native == "m7(9)" }.index
        assertEquals(
            listOf("C", "D", "Eb", "G", "Bb"),
            vocabulary.pitchClassNames(ninth, 0),
        )
    }
}

internal object AnimeChordTestAssets {
    fun locate(name: String): File {
        var directory: File? = File(System.getProperty("user.dir") ?: ".").absoluteFile
        while (directory != null) {
            val candidate = File(
                directory,
                "src/main/assets/${AnimeChordContract.ASSET_DIRECTORY}/$name",
            )
            if (candidate.isFile) return candidate
            val nested = File(
                directory,
                "PitchKit/PitchKit/src/main/assets/${AnimeChordContract.ASSET_DIRECTORY}/$name",
            )
            if (nested.isFile) return nested
            directory = directory.parentFile
        }
        throw AssertionError("Could not locate anime asset $name")
    }

    fun read(name: String): ByteArray = locate(name).readBytes()

    fun sha256(name: String): String = java.security.MessageDigest
        .getInstance("SHA-256")
        .digest(read(name))
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
