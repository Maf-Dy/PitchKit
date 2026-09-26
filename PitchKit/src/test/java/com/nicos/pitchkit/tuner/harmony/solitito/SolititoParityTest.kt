package com.nicos.pitchkit.tuner.harmony.solitito

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Kotlin solitito front end and decision layer against the Python reference.
 *
 * `tools/accuracy_audit/solitito_live.py` is the validated reimplementation of the whole
 * solitito serving path — it is what the Stage C part 2 screening
 * (`.accuracy-work/annotations/live-solitito-report.md`) was run with — so it is the spec
 * here, and `tools/accuracy_audit/export_solitito_fixture.py` freezes one 3 s GuitarSet
 * clip of it into `src/test/resources/solitito-parity.*`.
 *
 * Two claims, deliberately separated:
 *
 *  * **The front end matches.** Every one of the 156 frames' 168 features agrees with the
 *    reference to 1e-4 relative, and so does the raw frame RMS the gate and the attack
 *    detector read. That is the number that bounds how far the real model's inputs could
 *    drift from the ones the screening measured.
 *  * **The decision layer matches exactly.** The reference's own head logits are replayed
 *    through [SolititoDecisionLayer], so the tick schedule, the window fill, the gate, the
 *    confidence-weighted vote, the latch and the vocabulary are diffed with no ONNX
 *    session in the way — which also means this test runs under a plain
 *    `testDebugUnitTest`, where ONNX Runtime's native library will not load on Android
 *    Studio's JBR (see `LiveReplayTest.onnxRuntimeLoadsOnTheJvm`). The real graph is
 *    exercised by the JVM replay sweep on the patched JDK.
 *
 * The clip is from GuitarSet (Xi et al., ISMIR 2018), CC BY 4.0; see
 * `THIRD_PARTY_NOTICES.md`.
 */
class SolititoParityTest {

    private val fixture: JSONObject by lazy {
        JSONObject(resource("solitito-parity.json").toString(Charsets.UTF_8))
    }

    @Test
    fun fixtureIsTheOneThisTestWasWrittenFor() {
        assertEquals(1, fixture.getInt("schema_version"))
        assertEquals(SolititoContract.SAMPLE_RATE, fixture.getInt("target_rate"))
        assertEquals(SolititoContract.HOP_LENGTH, fixture.getInt("hop_length"))
        assertEquals(SolititoContract.FFT_SIZE, fixture.getInt("fft_size"))
        assertEquals(SolititoContract.CONTEXT_FRAMES, fixture.getInt("context_frames"))
        assertEquals(SolititoContract.FEATURE_COUNT, fixture.getInt("feature_count"))
        assertEquals(SolititoContract.GATE_DB, fixture.getDouble("gate_db"), 1e-12)
        assertEquals(
            "The fixture was generated from a different upstream model",
            SolititoContract.MODEL_SHA256,
            fixture.getString("model_sha256"),
        )
        assertEquals(
            "The fixture was generated from a different upstream dsp_weights.json",
            SolititoContract.UPSTREAM_WEIGHTS_SHA256,
            fixture.getString("dsp_weights_sha256"),
        )
        // Each binary is hashed, so a half-written or mismatched regeneration cannot
        // quietly weaken the comparison below.
        for ((name, key) in listOf(
            "solitito-parity-audio.s16" to "audio_sha256",
            "solitito-parity-features.f32" to "features_sha256",
            "solitito-parity-logits.f32" to "logits_sha256",
            "solitito-parity-rms.f32" to "rms_sha256",
        )) {
            assertEquals(name, fixture.getString(key), sha256(resource(name)))
        }
    }

    @Test
    fun resamplerReproducesTheReferenceGridWhateverTheChunking() {
        val audio = pcm()
        val rate = fixture.getInt("source_rate")
        val expected = fixture.getInt("resampled_sample_count")

        // One shot.
        val whole = SolititoLinearResampler().process(audio, rate)
        assertEquals("Resampled sample count", expected, whole.size)

        // ...and the geometry GuitarTunerListener actually delivers: 256 output samples
        // per capture buffer, which at 44.1 kHz is 706 source samples. The output must be
        // identical sample for sample and independent of the chunking, because the
        // position is recomputed from an absolute output index rather than accumulated.
        // 705 and 707 straddle the buffer that lands one output sample either side of a
        // hop, which is exactly where a mis-clamped internal drop shows up.
        for (chunk in listOf(706, 705, 707, 256, 1, 4096)) {
            val streamed = ArrayList<Float>(expected)
            val resampler = SolititoLinearResampler()
            var offset = 0
            while (offset < audio.size) {
                val size = minOf(chunk, audio.size - offset)
                for (value in resampler.process(audio.copyOfRange(offset, offset + size), rate)) {
                    streamed.add(value)
                }
                offset += size
            }
            assertEquals("Streamed sample count at chunk $chunk", whole.size, streamed.size)
            for (index in whole.indices) {
                assertEquals("chunk $chunk sample $index", whole[index], streamed[index], 0f)
            }
        }
    }

    @Test
    fun frontEndMatchesThePythonReferenceToOnePartInTenThousand() {
        val signal = SolititoLinearResampler().process(pcm(), fixture.getInt("source_rate"))
        val frames = fixture.getInt("frame_count")
        val expected = floats(resource("solitito-parity-features.f32"))
        val expectedRms = floats(resource("solitito-parity-rms.f32"))
        assertEquals(frames * SolititoContract.FEATURE_COUNT, expected.size)
        assertEquals(frames, expectedRms.size)

        val frontend = SolititoFrontend(plan(), gateDb = fixture.getDouble("gate_db"))
        val scratch = FloatArray(SolititoContract.FEATURE_COUNT)

        var worstFeature = 0.0
        var worstAt = -1
        var worstRms = 0.0
        var liveFrames = 0
        for (frame in 0 until frames) {
            val offset = frame * SolititoContract.HOP_LENGTH
            assertTrue(
                "Frame $frame runs past the resampled signal",
                offset + SolititoContract.FFT_SIZE <= signal.size,
            )
            val rms = frontend.frameRms(signal, offset)
            worstRms = maxOf(worstRms, relative(rms, expectedRms[frame]))
            val live = frontend.isLive(rms)
            if (live) liveFrames++
            java.util.Arrays.fill(scratch, 0f)
            if (live) frontend.features(signal, offset, scratch)
            for (bin in 0 until SolititoContract.FEATURE_COUNT) {
                val index = frame * SolititoContract.FEATURE_COUNT + bin
                val error = relative(scratch[bin], expected[index])
                if (error > worstFeature) {
                    worstFeature = error
                    worstAt = index
                }
            }
        }
        println(
            "solitito front end: frames=$frames live=$liveFrames " +
                "worst feature=$worstFeature at $worstAt worst rms=$worstRms"
        )
        assertEquals("Live frame count", fixture.getInt("live_frames"), liveFrames)
        assertTrue("Worst relative RMS error $worstRms", worstRms <= 1e-4)
        assertTrue(
            "Worst relative feature error $worstFeature at index $worstAt",
            worstFeature <= 1e-4,
        )

        // Vacuity guard: the same comparison run against the time-reversed signal has to
        // fail by a wide margin, or it would pass for any spectrogram at all.
        val reversed = FloatArray(signal.size) { signal[signal.size - 1 - it] }
        java.util.Arrays.fill(scratch, 0f)
        frontend.features(reversed, 0, scratch)
        var control = 0.0
        for (bin in 0 until SolititoContract.FEATURE_COUNT) {
            control = maxOf(control, relative(scratch[bin], expected[bin]))
        }
        assertTrue("Time-reversed control error $control is not large", control > 1e-2)
    }

    @Test
    fun decisionLayerReproducesTheEmittedLabelSequenceExactly() {
        val signal = SolititoLinearResampler().process(pcm(), fixture.getInt("source_rate"))
        val frames = fixture.getInt("frame_count")
        val logits = floats(resource("solitito-parity-logits.f32"))
        val headWidth = fixture.getInt("root_classes") + fixture.getInt("quality_classes") +
            fixture.getInt("pitch_classes")
        assertEquals(fixture.getInt("asked_count") * headWidth, logits.size)

        // The reference's own head logits, handed back in the order they were produced.
        var nextRow = 0
        val model = SolititoDecisionLayer.Model { _ ->
            val base = nextRow++ * headWidth
            val root = FloatArray(SolititoContract.ROOT_CLASSES) { logits[base + it] }
            val quality = FloatArray(SolititoContract.QUALITY_CLASSES) {
                logits[base + SolititoContract.ROOT_CLASSES + it]
            }
            val (rootIndex, rootConfidence) = SolititoContract.argmaxSoftmax(root)
            val (qualityIndex, qualityConfidence) = SolititoContract.argmaxSoftmax(quality)
            val name = SolititoVocabulary.QUALITIES[qualityIndex]
            val noise = rootIndex >= 12 || name == "N"
            SolititoDecisionLayer.Reading(
                rootIndex = rootIndex,
                quality = name,
                confidence = if (noise) 0.0 else rootConfidence * qualityConfidence,
            )
        }

        val frontend = SolititoFrontend(plan(), gateDb = fixture.getDouble("gate_db"))
        val decisions = SolititoDecisionLayer(model)
        val scratch = FloatArray(SolititoContract.FEATURE_COUNT)
        val zero = FloatArray(SolititoContract.FEATURE_COUNT)
        val actual = mutableListOf<SolititoDecisionLayer.Decision>()
        for (frame in 0 until frames) {
            val offset = frame * SolititoContract.HOP_LENGTH
            val rms = frontend.frameRms(signal, offset)
            val live = frontend.isLive(rms)
            if (live) frontend.features(signal, offset, scratch)
            decisions.push(if (live) scratch else zero, live, rms)?.let { actual.add(it) }
        }

        val ticks = fixture.getJSONArray("ticks")
        assertEquals("Tick count", ticks.length(), actual.size)
        assertEquals("Windows the model was asked about", fixture.getInt("asked_count"), nextRow)

        val expectedLabels = mutableListOf<String?>()
        val actualLabels = mutableListOf<String?>()
        for (index in 0 until ticks.length()) {
            val expected = ticks.getJSONObject(index)
            val row = actual[index]
            // audio_s is the time of the last sample in the analysis window, which is the
            // clock the reference's trace and the `.lab` files share.
            val expectedFrame = Math.round(
                (expected.getDouble("audio_s") * SolititoContract.SAMPLE_RATE -
                    SolititoContract.FFT_SIZE) / SolititoContract.HOP_LENGTH
            )
            assertEquals("tick $index frame", expectedFrame, row.frameIndex)
            assertEquals("tick $index asked", expected.getBoolean("asked"), row.asked)
            assertEquals("tick $index fill", expected.getDouble("fill"), row.fill, 5e-5)
            if (expected.getBoolean("asked")) {
                assertEquals(
                    "tick $index raw chord",
                    expected.getString("raw_chord"),
                    row.rawChord,
                )
                assertEquals("tick $index quality", expected.getString("quality"), row.quality)
                assertEquals("tick $index voted", expected.getString("voted"), row.voted)
                assertEquals(
                    "tick $index latched",
                    if (expected.isNull("latched")) null else expected.getString("latched"),
                    row.latched,
                )
                assertEquals("tick $index onset", expected.getInt("onset_id"), row.onsetId)
                assertEquals(
                    "tick $index model confidence",
                    expected.getDouble("model_confidence"),
                    row.modelConfidence!!,
                    1e-5,
                )
                assertEquals(
                    "tick $index emitted confidence",
                    expected.getDouble("confidence"),
                    row.confidence,
                    1e-5,
                )
            } else {
                assertNull("tick $index emitted nothing", row.emitted)
            }
            expectedLabels.add(if (expected.isNull("emitted")) null else expected.getString("emitted"))
            actualLabels.add(row.emitted)
        }
        assertEquals("Emitted label sequence", expectedLabels, actualLabels)

        val counters = fixture.getJSONObject("counters")
        assertEquals("inferences", counters.getLong("inferences"), decisions.inferences)
        assertEquals("skipped_low_fill", counters.getLong("skipped_low_fill"), decisions.skippedLowFill)
        assertEquals("not_named", counters.getLong("not_named"), decisions.notNamed)
        assertEquals("note", counters.getLong("note"), decisions.noteReadings)
        assertEquals("noise", counters.getLong("noise"), decisions.noiseReadings)
        assertEquals("live_frames", counters.getLong("live_frames"), decisions.liveFrames)
        assertEquals("gated_frames", counters.getLong("gated_frames"), decisions.gatedFrames)
    }

    // ------------------------------------------------------------------ fixtures

    private fun pcm(): FloatArray {
        val bytes = resource("solitito-parity-audio.s16")
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val samples = FloatArray(buffer.remaining())
        // WavReader decodes 16-bit PCM as `short / 32768f`, and so does the reference's
        // `read_wav_mono`; the fixture stores the file's own int16 so both agree exactly.
        for (index in samples.indices) samples[index] = buffer.get(index) / 32768f
        assertEquals(fixture.getInt("clip_sample_count"), samples.size)
        return samples
    }

    private fun plan(): SolititoDspPlan =
        SolititoDspPlanDecoder.decodeAndVerify(SolititoTestAssets.plan())

    private fun relative(actual: Float, expected: Float): Double {
        val difference = abs(actual.toDouble() - expected.toDouble())
        if (difference == 0.0) return 0.0
        return difference / maxOf(abs(expected.toDouble()), 1e-3)
    }

    private fun resource(name: String): ByteArray =
        requireNotNull(javaClass.getResourceAsStream("/$name")) { "missing test resource $name" }
            .use { it.readBytes() }

    private fun floats(bytes: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}

/** Walks up to `src/main/assets/solitito`, the same way the ChordFormer tests do. */
internal object SolititoTestAssets {
    fun locate(name: String): File? {
        var cursor: File? = File(System.getProperty("user.dir") ?: ".").absoluteFile
        while (cursor != null) {
            for (prefix in listOf("src/main/assets", "PitchKit/PitchKit/src/main/assets")) {
                val candidate = File(
                    cursor,
                    "$prefix/${SolititoContract.ASSET_DIRECTORY}/$name",
                )
                if (candidate.isFile) return candidate
            }
            cursor = cursor.parentFile
        }
        return null
    }

    fun plan(): ByteArray = requireNotNull(locate(SolititoContract.PLAN_FILE)) {
        "solitito DSP plan is not installed; run " +
            "tools/accuracy_audit/export_solitito_assets.py"
    }.readBytes()
}
