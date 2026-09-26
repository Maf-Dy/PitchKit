package com.nicos.pitchkit.tuner.harmony.consonance

import java.io.File
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

/**
 * Frame-by-frame parity with tools/accuracy_audit/consonance_dictionary_decode.py.
 * Each fixture holds one saved 862-frame forward pass plus the label sequence the Python
 * decoder produces for it over the exported dictionary, for one selectable lane
 * (likelihood + chord-change penalty). The lanes the app offers must all decode
 * label-for-label like their reference, not only the asset's own default.
 */
class ConsonanceDictionaryViterbiParityTest {
    @Test
    fun matchesThePythonReferenceFrameByFrame() {
        assertLaneParity(
            "/consonance-viterbi-fixture.json",
            ConsonanceLikelihood.COMPETITIVE,
            expectedPenalty = ConsonanceContract.TRANSITION_PENALTY,
        )
        // The asset's own declaration is what the default lane runs.
        val dictionary = parseDictionary()
        assertEquals(ConsonanceLikelihood.COMPETITIVE, dictionary.likelihood)
        assertEquals(ConsonanceContract.TRANSITION_PENALTY, dictionary.transitionPenalty, 0.0)
    }

    @Test
    fun matchesThePythonBernoulliReferenceFrameByFrame() {
        assertLaneParity(
            "/consonance-viterbi-fixture-bernoulli-p100.json",
            ConsonanceLikelihood.BERNOULLI,
            expectedPenalty = 100.0,
        )
    }

    @Test
    fun matchesThePythonCompetitiveBassReferenceFrameByFrame() {
        assertLaneParity(
            "/consonance-viterbi-fixture-competitive-bass-p100.json",
            ConsonanceLikelihood.COMPETITIVE_BASS,
            expectedPenalty = 100.0,
        )
    }

    /**
     * A lane that always decoded like competitive would make its fixture vacuous, so the
     * Bernoulli lane is checked to actually diverge on this forward pass. The
     * competitive-bass lane is deliberately not asserted to differ here: it only changes
     * root-position candidates whose bass head prefers another chord tone, which this
     * particular clip never triggers (its fixture is identical to the competitive one).
     * ConsonanceDictionaryDecoderTest covers that rule directly.
     */
    @Test
    fun theBernoulliLaneDivergesFromCompetitiveOnTheSameForwardPass() {
        val competitive = decodeFixture(
            "/consonance-viterbi-fixture.json",
            ConsonanceLikelihood.COMPETITIVE,
            ConsonanceContract.TRANSITION_PENALTY,
        )
        val bernoulli = decodeFixture(
            "/consonance-viterbi-fixture-bernoulli-p100.json",
            ConsonanceLikelihood.BERNOULLI,
            100.0,
        )
        assertNotEquals(competitive, bernoulli)
    }

    private fun assertLaneParity(
        resource: String,
        likelihood: ConsonanceLikelihood,
        expectedPenalty: Double,
    ) {
        val fixture = readFixture(resource)
        assertEquals(1, fixture.getInt("schema_version"))
        assertEquals(likelihood.id, fixture.getString("likelihood"))
        assertEquals(expectedPenalty, fixture.getDouble("transition_penalty"), 0.0)

        val dictionaryBytes = locateAsset(ConsonanceContract.DICTIONARY_FILE).readBytes()
        assertEquals(ConsonanceContract.DICTIONARY_SHA256, sha256(dictionaryBytes))
        assertEquals(fixture.getString("dictionary_sha256"), sha256(dictionaryBytes))

        val actual = decodeFixture(resource, likelihood, expectedPenalty)
        val expected = fixture.getJSONArray("labels")
        assertEquals(expected.length(), actual.size)
        assertEquals(List(expected.length()) { expected.getString(it) }, actual)
    }

    private fun decodeFixture(
        resource: String,
        likelihood: ConsonanceLikelihood,
        penalty: Double,
    ): List<String> {
        val fixture = readFixture(resource)
        val dictionary = parseDictionary()
        val frames = fixture.getInt("frames")
        val heads = ConsonanceHeads(
            frames = frames,
            root = floats(fixture.getJSONArray("root")),
            bass = floats(fixture.getJSONArray("bass")),
            pitch = floats(fixture.getJSONArray("pitch")),
        )
        assertEquals(frames * ConsonanceContract.ROOT_COUNT, heads.root.size)
        val decoded = ConsonanceDictionaryDecoder(
            dictionary = dictionary,
            transitionPenalty = penalty,
            likelihood = likelihood,
        ).decodeFrames(heads)
        return decoded.map { dictionary.candidates[it].rawLabel }
    }

    private fun parseDictionary(): ConsonanceDictionary = ConsonanceDictionaryParser.parse(
        locateAsset(ConsonanceContract.DICTIONARY_FILE).readBytes().toString(Charsets.UTF_8),
        preferFlats = false,
    )

    private fun readFixture(resource: String): JSONObject = JSONObject(
        (javaClass.getResourceAsStream(resource) ?: error("Missing fixture $resource"))
            .use { it.readBytes() }
            .toString(Charsets.UTF_8)
    )

    private fun floats(array: JSONArray): FloatArray =
        FloatArray(array.length()) { array.getDouble(it).toFloat() }

    private fun locateAsset(name: String): File {
        var directory: File? = File(System.getProperty("user.dir") ?: ".").absoluteFile
        while (directory != null) {
            val candidate = File(
                directory,
                "src/main/assets/${ConsonanceContract.ASSET_DIR}/$name",
            )
            if (candidate.isFile) return candidate
            val nested = File(
                directory,
                "PitchKit/PitchKit/src/main/assets/${ConsonanceContract.ASSET_DIR}/$name",
            )
            if (nested.isFile) return nested
            directory = directory.parentFile
        }
        throw AssertionError("Could not locate Consonance asset $name")
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
