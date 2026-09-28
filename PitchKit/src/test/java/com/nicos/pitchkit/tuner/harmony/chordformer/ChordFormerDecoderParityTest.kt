package com.nicos.pitchkit.tuner.harmony.chordformer

import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaDictionaryDecoder
import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaDictionaryParser
import com.nicos.pitchkit.tuner.harmony.lvchordia.LvChordiaHeads
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Frame-by-frame parity with `extractors/xhmm_ismir.py::XHMMDecoder`.
 *
 * onnxruntime has no JVM build here, so the fixture holds the real thing the Kotlin
 * decoder would have been handed: 512 committed frames of onnxruntime head
 * probabilities from a real 2048-frame ChordFormer window, plus the label sequence the
 * Python decoder produces for exactly those frames over the exported `submission`
 * dictionary. Written by `tools/accuracy_audit/export_chordformer.py --stage verify`.
 */
class ChordFormerDecoderParityTest {
    @Test
    fun matchesThePythonXhmmFrameByFrame() {
        val stream = javaClass.getResourceAsStream("/chordformer-decoder-fixture.json")
        org.junit.Assume.assumeTrue(
            "The ChordFormer decoder fixture is written locally by export_chordformer.py",
            stream != null,
        )
        val fixture = JSONObject(
            stream!!
                .use { it.readBytes() }
                .toString(Charsets.UTF_8)
        )
        assertEquals(1, fixture.getInt("schema_version"))
        assertEquals(ChordFormerContract.VOCABULARY, fixture.getString("vocabulary"))
        assertEquals(
            ChordFormerContract.TRANSITION_PENALTY,
            fixture.getDouble("transition_penalty"),
            0.0,
        )
        assertEquals(ChordFormerContract.MODEL_SHA256, fixture.getString("model_sha256"))

        val dictionaryBytes = ChordFormerTestAssets
            .locate(ChordFormerContract.DICTIONARY_FILE)
            .readBytes()
        assertEquals(ChordFormerContract.DICTIONARY_SHA256, sha256(dictionaryBytes))
        assertEquals(fixture.getString("dictionary_sha256"), sha256(dictionaryBytes))

        val frames = fixture.getInt("frames")
        val heads = fixture.getJSONObject("heads")
        val bundle = LvChordiaHeads(
            frames = frames,
            triad = floats(heads.getJSONArray("triad")),
            bass = floats(heads.getJSONArray("bass")),
            seventh = floats(heads.getJSONArray("seventh")),
            ninth = floats(heads.getJSONArray("ninth")),
            eleventh = floats(heads.getJSONArray("eleventh")),
            thirteenth = floats(heads.getJSONArray("thirteenth")),
        )
        assertEquals(frames * ChordFormerContract.TRIAD_COUNT, bundle.triad.size)
        assertEquals(frames * ChordFormerContract.BASS_COUNT, bundle.bass.size)
        assertEquals(frames * ChordFormerContract.SEVENTH_COUNT, bundle.seventh.size)
        assertEquals(frames * ChordFormerContract.NINTH_COUNT, bundle.ninth.size)
        assertEquals(frames * ChordFormerContract.ELEVENTH_COUNT, bundle.eleventh.size)
        assertEquals(frames * ChordFormerContract.THIRTEENTH_COUNT, bundle.thirteenth.size)

        val dictionary = LvChordiaDictionaryParser.parse(
            dictionaryBytes.toString(Charsets.UTF_8),
            preferFlats = false,
        )
        assertEquals(
            ChordFormerContract.TRANSITION_PENALTY,
            dictionary.transitionPenalty,
            0.0,
        )

        val decoded = LvChordiaDictionaryDecoder(dictionary).decode(bundle)
        assertEquals(frames, decoded.size)

        // The decoder reports display spellings; raw Harte labels are unique, so map the
        // Python reference forward rather than mapping the decoder's answer back.
        val rawToDisplay = dictionary.candidates.associateBy({ it.rawLabel }, { it.displayLabel })
        val expected = fixture.getJSONArray("labels")
        val expectedDisplay = List(expected.length()) { rawToDisplay.getValue(expected.getString(it)) }
        val actualDisplay = decoded.map { it.label }
        assertEquals(expectedDisplay.size, actualDisplay.size)
        assertEquals(expectedDisplay, actualDisplay)

        // A fixture that decoded to a single chord would pass vacuously.
        assertTrue(
            "Distinct labels: ${actualDisplay.distinct()}",
            actualDisplay.distinct().size >= 3,
        )
    }

    @Test
    fun everyDictionaryCandidateHasADisplaySpelling() {
        val dictionary = LvChordiaDictionaryParser.parse(
            ChordFormerTestAssets.locate(ChordFormerContract.DICTIONARY_FILE)
                .readBytes()
                .toString(Charsets.UTF_8),
            preferFlats = false,
        )
        // 25 chord types x 12 roots + N, the `submission` vocabulary.
        assertEquals(301, dictionary.candidates.size)
        assertEquals("N", dictionary.candidates.first().rawLabel)

        val unformatted = dictionary.candidates
            .drop(1)
            .filter { it.displayLabel.isNullOrBlank() || it.displayLabel == it.rawLabel }
        assertTrue("Unformatted labels: ${unformatted.map { it.rawLabel }}", unformatted.isEmpty())

        val duplicates = dictionary.candidates
            .groupBy { it.displayLabel }
            .filterValues { it.size > 1 }
            .keys
        assertTrue("Ambiguous display labels: $duplicates", duplicates.isEmpty())

        for (candidate in dictionary.candidates) {
            assertTrue(candidate.rawLabel, candidate.triad in 0 until ChordFormerContract.TRIAD_COUNT)
            assertTrue(candidate.rawLabel, candidate.bass + 1 in 0 until ChordFormerContract.BASS_COUNT)
            assertTrue(candidate.rawLabel, candidate.seventh < ChordFormerContract.SEVENTH_COUNT)
            assertTrue(candidate.rawLabel, candidate.ninth < ChordFormerContract.NINTH_COUNT)
            assertTrue(candidate.rawLabel, candidate.eleventh < ChordFormerContract.ELEVENTH_COUNT)
            assertTrue(candidate.rawLabel, candidate.thirteenth < ChordFormerContract.THIRTEENTH_COUNT)
        }
    }

    private fun floats(array: JSONArray): FloatArray =
        FloatArray(array.length()) { array.getDouble(it).toFloat() }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
