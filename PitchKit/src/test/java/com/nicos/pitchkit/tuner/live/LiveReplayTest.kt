package com.nicos.pitchkit.tuner.live

import ai.onnxruntime.OrtEnvironment
import com.nicos.pitchkit.tuner.harmony.ChordRecognizer
import com.nicos.pitchkit.tuner.harmony.btc.BtcContract
import com.nicos.pitchkit.tuner.harmony.btc.BtcMetadata
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetContract
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetStreamingRecognizer
import com.nicos.pitchkit.tuner.harmony.solitito.SolititoContract
import com.nicos.pitchkit.tuner.models.AudioFrame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeNoException
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * JUnit entry point for the live replay harness.
 *
 * The replay itself only runs when `-Dlive.replay.manifest=<file>` is supplied,
 * so a plain `testDebugUnitTest` stays green and fast. The two other tests are
 * the build-plumbing guards: that desktop ONNX Runtime loads on this JVM, and
 * that the WAV reader agrees with itself on both corpus subtypes. The ONNX guard
 * assumes out rather than failing when the JVM's own bundled MSVC runtime is too
 * old for the native library, which is the case for Android Studio's JBR.
 */
class LiveReplayTest {

    /** The asset sub-directory ChordNetAndroidFactory reads on the device. */
    private val chordNetAssetDir = "chordnet"

    @Test
    fun onnxRuntimeLoadsOnTheJvm() {
        // Android Studio's JBR bundles MSVC runtime 14.29 next to java.exe, and
        // Windows resolves msvcp140.dll from the executable's own directory
        // first, so onnxruntime.dll (which needs 14.4x or newer) fails its
        // DllMain there with ERROR_DLL_INIT_FAILED. That is a property of the
        // JVM, not of this module, so skip rather than fail; the live replay
        // sweep runs on the patched JDK that tools/accuracy_audit/live_replay.py
        // prepares. See .accuracy-work/annotations/live-stage-a-report.md.
        val environment = try {
            OrtEnvironment.getEnvironment()
        } catch (error: Throwable) {
            assumeNoException(
                "ONNX Runtime has no usable native library on this JVM " +
                    "(${System.getProperty("java.home")}); run the live sweep on an " +
                    "ONNX-capable JDK - see live_replay.py --prepare-jvm",
                error,
            )
            return
        }
        assertNotNull("OrtEnvironment.getEnvironment() returned null on the JVM", environment)
        assertSame("OrtEnvironment is not process-wide", environment, OrtEnvironment.getEnvironment())
    }

    @Test
    fun wavReaderRoundTripsPcm16AndFloat32() {
        val expected = FloatArray(512) { index -> Math.sin(index * 0.05).toFloat() * 0.5f }

        val pcm16 = WavReader.read(writeTemp("pcm16", wav(expected, 22_050, 1, float = false)))
        assertEquals(22_050, pcm16.sampleRate)
        assertEquals(1, pcm16.channelCount)
        assertEquals(expected.size, pcm16.frameCount)
        for (index in expected.indices) {
            assertEquals(expected[index].toDouble(), pcm16.samples[index].toDouble(), 1.0 / 16384)
        }

        val float32 = WavReader.read(writeTemp("float32", wav(expected, 44_100, 1, float = true)))
        assertEquals(44_100, float32.sampleRate)
        for (index in expected.indices) {
            assertEquals(expected[index].toDouble(), float32.samples[index].toDouble(), 1e-7)
        }

        // Stereo is averaged down exactly like AudioFrame.toMono().
        val stereo = FloatArray(expected.size * 2) { if (it % 2 == 0) expected[it / 2] else 0f }
        val read = WavReader.read(writeTemp("stereo", wav(stereo, 44_100, 2, float = true)))
        assertEquals(2, read.channelCount)
        assertEquals(expected.size, read.frameCount)
        val mono = read.toMono()
        for (index in expected.indices) {
            assertEquals(expected[index] / 2.0, mono[index].toDouble(), 1e-7)
        }
    }

    /**
     * The BTC lane is the phone's fourth live engine and the only one AUTO never
     * picks, so nothing else in the suite would notice if its replay geometry or
     * its on-disk assets drifted away from what `GuitarTunerListener` hands it.
     *
     * `GuitarTunerListener.kt` gives a `BtcStreamingRecognizer` `BtcContract`'s
     * own sample rate and hop as `AudioCapture`'s negotiated rate and buffer
     * size; `LiveReplay.Engine.BTC` must carry exactly those, or the replay would
     * be feeding a geometry the device never uses. The asset digests are the same
     * two the recognizer checks at construction, asserted here so a swapped model
     * or plan fails fast rather than part-way through a sweep.
     */
    @Test
    fun btcEngineMatchesTheShippedLiveGeometry() {
        val engine = LiveReplay.Engine.of("btc")
        assertEquals(LiveReplay.Engine.BTC, engine)
        assertEquals(BtcContract.SAMPLE_RATE, engine.sampleRate)
        assertEquals(BtcContract.HOP_LENGTH, engine.bufferSize)
        // Same 92.9 ms frame period as ChordNet: 2048 @ 22 050 Hz.
        assertEquals(2_048.0 / 22_050, engine.framePeriodSeconds, 1e-12)

        val model = btcAsset(BtcContract.MODEL_FILE)
        val meta = btcAsset(BtcContract.METADATA_FILE)
        val plan = btcAsset(BtcContract.PLAN_FILE)
        assumeTrue(
            "BTC assets are not installed; run tools/fetch-btc.ps1",
            model != null && meta != null && plan != null,
        )
        val metadata = BtcMetadata.parse(meta!!.readBytes())
        assertEquals(
            "btc.onnx does not match the SHA-256 in btc-meta.properties",
            metadata.modelSha256,
            LiveReplay.sha256(model!!.readBytes()),
        )
        assertEquals(
            "BTC CQT plan does not match BtcContract.PLAN_SHA256",
            BtcContract.PLAN_SHA256,
            LiveReplay.sha256(plan!!.readBytes()),
        )
    }

    /**
     * The rescue ablation lanes (`.accuracy-work/annotations/live-rescue-ablation-report.md`).
     *
     * Each `-norescue` / `-rescue` / `-rescue60` variant must be the *same engine*
     * as the lane it ablates - same assets, same negotiated rate, same buffer -
     * differing only in the two arguments handed to the shipped recognizer. If a
     * variant's geometry ever drifted from its base, the ablation would be
     * comparing two different experiments and the report's deltas would mean
     * nothing.
     *
     * The shipped lanes are pinned to the shipped configuration in the same test,
     * so an accidental flip of a default shows up here rather than in a
     * ten-minute sweep. Since the ablation, that configuration is **rescue off**;
     * the `-rescue` variants carry the pre-ablation one at 0.80.
     */
    @Test
    fun rescueAblationVariantsAreTheirBaseEngineWithOneKnobMoved() {
        val base = mapOf(
            "chordnet-norescue" to LiveReplay.Engine.CHORD_NET,
            "chordnet-rescue" to LiveReplay.Engine.CHORD_NET,
            "chordnet-rescue60" to LiveReplay.Engine.CHORD_NET,
            "crema-norescue" to LiveReplay.Engine.CREMA,
            "crema-rescue" to LiveReplay.Engine.CREMA,
            "btc-norescue" to LiveReplay.Engine.BTC,
            "btc-rescue" to LiveReplay.Engine.BTC,
        )
        for ((id, parent) in base) {
            val engine = LiveReplay.Engine.of(id)
            assertEquals(id + " sample rate", parent.sampleRate, engine.sampleRate)
            assertEquals(id + " buffer size", parent.bufferSize, engine.bufferSize)
            assertEquals(
                id + " frame period",
                parent.framePeriodSeconds,
                engine.framePeriodSeconds,
                1e-12,
            )
        }

        // The shipped lanes: rescue OFF since the ablation, and the threshold
        // they would use if it were ever switched back on is 0.60, not 0.80.
        for (engine in listOf(
            LiveReplay.Engine.CHORD_NET,
            LiveReplay.Engine.CREMA,
            LiveReplay.Engine.BTC,
        )) {
            assertFalse(engine.id + " must ship with the rescue off", engine.dspRescue)
            assertEquals(
                engine.id + " must default to the 0.60 gate",
                LiveReplay.Engine.DEFAULT_RESCUE_THRESHOLD,
                engine.rescueConfidenceThreshold,
                1e-12,
            )
        }

        // The ablation knobs themselves.
        assertFalse(LiveReplay.Engine.of("chordnet-norescue").dspRescue)
        assertFalse(LiveReplay.Engine.of("crema-norescue").dspRescue)
        assertFalse(LiveReplay.Engine.of("btc-norescue").dspRescue)
        // The pre-ablation behaviour, still benchmarkable: rescue on at 0.80.
        for (id in listOf("chordnet-rescue", "crema-rescue", "btc-rescue")) {
            val legacy = LiveReplay.Engine.of(id)
            assertTrue(id + " must carry the rescue", legacy.dspRescue)
            assertEquals(
                id + " must gate at the pre-ablation 0.80",
                LiveReplay.Engine.LEGACY_RESCUE_THRESHOLD,
                legacy.rescueConfidenceThreshold,
                1e-12,
            )
        }
        assertTrue(LiveReplay.Engine.of("chordnet-rescue60").dspRescue)
        assertEquals(
            0.60,
            LiveReplay.Engine.of("chordnet-rescue60").rescueConfidenceThreshold,
            1e-12,
        )
        // Classic DSP has no rescue lane to ablate.
        assertFalse(LiveReplay.Engine.CLASSIC.dspRescue)
    }

    /**
     * The production seam, exercised on the real ChordNet lane.
     *
     * Two claims the ablation report rests on, neither observable from the enum:
     *
     *  * **The default is rescue-off.** A recognizer built with no rescue
     *    arguments and one built with `dspRescue = false` spelled out emit the
     *    identical label and backend sequence on identical audio, and neither
     *    emits a `+ temporal CQT DSP` backend - the only externally visible
     *    evidence that `CqtChordTemplateDetector` overruled the model.
     *  * **`dspRescue = true` still restores the lane**, at the same row count,
     *    so the pre-ablation configuration the report baselines against remains
     *    reachable and both backend spellings stay pinned.
     *
     * Assumed out when the assets are absent or ONNX Runtime has no usable native
     * library on this JVM, the same as `onnxRuntimeLoadsOnTheJvm`.
     */
    @Test
    fun chordNetDefaultsToRescueOffAndTheLaneReturnsWhenEnabled() {
        val model = asset(chordNetAssetDir, ChordNetContract.MODEL_FILE)
        val plan = asset(chordNetAssetDir, ChordNetContract.PLAN_FILE)
        assumeTrue("ChordNet assets are not installed", model != null && plan != null)
        try {
            OrtEnvironment.getEnvironment()
        } catch (error: Throwable) {
            assumeNoException("ONNX Runtime has no usable native library on this JVM", error)
            return
        }
        val modelBytes = model!!.readBytes()
        val planBytes = plan!!.readBytes()
        val audio = triad(seconds = 4.0, rate = ChordNetContract.SAMPLE_RATE)

        val implicitDefault = drive(
            ChordNetStreamingRecognizer(modelBytes = modelBytes, planBytes = planBytes),
            audio,
        )
        val explicitDefault = drive(
            ChordNetStreamingRecognizer(
                modelBytes = modelBytes,
                planBytes = planBytes,
                dspRescue = false,
                rescueConfidenceThreshold =
                    ChordNetStreamingRecognizer.DSP_OVERRIDE_MODEL_CONFIDENCE,
            ),
            audio,
        )
        assertEquals(
            "Spelling the rescue arguments out at their defaults changed the output",
            implicitDefault,
            explicitDefault,
        )
        // The default is the ablated configuration: the lane cannot appear.
        for (row in implicitDefault) {
            assertFalse("The default emitted a DSP-rescued row: " + row, row.contains("DSP"))
        }
        assertEquals(
            "The default gate is 0.60 since the rescue ablation",
            0.60,
            ChordNetStreamingRecognizer.DSP_OVERRIDE_MODEL_CONFIDENCE,
            1e-12,
        )

        val legacyRescue = drive(
            ChordNetStreamingRecognizer(
                modelBytes = modelBytes,
                planBytes = planBytes,
                dspRescue = true,
                rescueConfidenceThreshold = 0.80,
            ),
            audio,
        )
        assertEquals(
            "The rescue-on run must produce one row per fed buffer, like the default",
            implicitDefault.size,
            legacyRescue.size,
        )
        // Rows are `label|backend`; the backend string is what the trace scorer
        // attributes the rescue on, so pin its two spellings.
        for (row in legacyRescue) {
            val backend = row.substringAfter('|', "")
            assertTrue(
                "Unexpected ChordNet backend: " + row,
                row == "-" ||
                    backend == "ChordNet 2E1D" ||
                    backend == "ChordNet + temporal CQT DSP",
            )
        }
    }

    /** `label|backend` per fed buffer, so a diff shows which buffer diverged. */
    private fun drive(recognizer: ChordRecognizer, audio: FloatArray): List<String> {
        val rows = mutableListOf<String>()
        recognizer.use { engine ->
            var offset = 0
            while (offset < audio.size) {
                val size = minOf(ChordNetContract.HOP_LENGTH, audio.size - offset)
                val chunk = audio.copyOfRange(offset, offset + size)
                offset += size
                val recognition = engine.recognize(
                    AudioFrame(chunk, ChordNetContract.SAMPLE_RATE, 1)
                )
                rows += if (recognition == null) {
                    "-"
                } else {
                    recognition.label + "|" + recognition.backend
                }
            }
        }
        return rows
    }

    /** A sustained C major triad with a few harmonics; deterministic by construction. */
    private fun triad(seconds: Double, rate: Int): FloatArray {
        val fundamentals = doubleArrayOf(261.626, 329.628, 391.995)
        val samples = FloatArray((seconds * rate).toInt())
        for (index in samples.indices) {
            val t = index.toDouble() / rate
            var value = 0.0
            for (fundamental in fundamentals) {
                for (harmonic in 1..4) {
                    value += Math.sin(2 * Math.PI * fundamental * harmonic * t) / (harmonic * harmonic)
                }
            }
            // A short attack ramp so the gesture accumulator sees a real onset.
            val envelope = Math.min(1.0, t / 0.08)
            samples[index] = (value * 0.12 * envelope).toFloat()
        }
        return samples
    }

    /** Same upward walk the ChordFormer parity tests use; null when not installed. */
    private fun asset(directory: String, name: String): File? {
        var cursor: File? = File(System.getProperty("user.dir") ?: ".").absoluteFile
        while (cursor != null) {
            for (prefix in listOf("src/main/assets", "PitchKit/PitchKit/src/main/assets")) {
                val candidate = File(cursor, prefix + "/" + directory + "/" + name)
                if (candidate.isFile) return candidate
            }
            cursor = cursor.parentFile
        }
        return null
    }

    private fun btcAsset(name: String): File? = asset(BtcContract.ASSET_DIR, name)

    /**
     * The solitito-ai lane and its one-constant gate ablation.
     *
     * `GuitarTunerListener` gives a `SolititoStreamingRecognizer` 16 kHz and a
     * 256-sample buffer, which is a 16 ms frame period and the smallest input
     * latency term in the APK; the replay must feed exactly that geometry. The
     * `-gate34` variant exists so the port can reproduce *both* sides of the
     * screening's headline result - upstream's own -34 dBFS gate and the -60 the
     * port ships - from the same code path, and it must differ from the shipped
     * lane in nothing but that constant.
     */
    @Test
    fun solititoEngineMatchesTheShippedLiveGeometryAndItsGateAblation() {
        val engine = LiveReplay.Engine.of("solitito")
        assertEquals(LiveReplay.Engine.SOLITITO, engine)
        assertEquals(SolititoContract.SAMPLE_RATE, engine.sampleRate)
        assertEquals(SolititoContract.HOP_LENGTH, engine.bufferSize)
        assertEquals(256.0 / 16_000, engine.framePeriodSeconds, 1e-12)
        assertEquals(SolititoContract.GATE_DB, engine.gateDb!!, 1e-12)
        // solitito has no CqtChordTemplateDetector rescue lane to ablate.
        assertFalse(engine.dspRescue)
        // It resamples everything itself, so it always wants the least pre-damaged
        // rendering rather than one matched to its own rate.
        assertTrue(engine.preferHighestRate)

        val ablation = LiveReplay.Engine.of("solitito-gate34")
        assertEquals(engine.sampleRate, ablation.sampleRate)
        assertEquals(engine.bufferSize, ablation.bufferSize)
        assertEquals(engine.dspRescue, ablation.dspRescue)
        assertEquals(engine.preferHighestRate, ablation.preferHighestRate)
        assertEquals(SolititoContract.SHIPPED_GATE_DB, ablation.gateDb!!, 1e-12)

        val model = asset(SolititoContract.ASSET_DIRECTORY, SolititoContract.MODEL_FILE)
        val plan = asset(SolititoContract.ASSET_DIRECTORY, SolititoContract.PLAN_FILE)
        assumeTrue(
            "solitito assets are not installed; run " +
                "tools/accuracy_audit/export_solitito_assets.py",
            model != null && plan != null,
        )
        assertEquals(
            "solitito-v2-take6.onnx does not match SolititoContract.MODEL_SHA256",
            SolititoContract.MODEL_SHA256,
            LiveReplay.sha256(model!!.readBytes()),
        )
        assertEquals(
            "solitito-dsp.bin does not match SolititoContract.PLAN_SHA256",
            SolititoContract.PLAN_SHA256,
            LiveReplay.sha256(plan!!.readBytes()),
        )
    }

    @Test
    fun replaysLiveManifest() {
        val manifest = System.getProperty("live.replay.manifest").orEmpty()
        assumeTrue(
            "Set -Dlive.replay.manifest=<manifest.json> to run the live replay harness",
            manifest.isNotBlank(),
        )
        val file = File(manifest)
        assertTrue("Manifest not found: ${file.absolutePath}", file.isFile)

        val results = LiveReplay.runManifest(file)
        assertTrue("Replay produced no traces", results.isNotEmpty())
        for (result in results) {
            assertTrue("Empty trace for ${result.engine.id}/${result.item.id}", result.rows > 0)
            assertTrue("Missing trace ${result.trace}", result.trace.isFile)
        }

        val summary = LiveReplay.summary(results)
        System.out.println("LIVE_REPLAY_SUMMARY $summary")
        System.getProperty("live.replay.summary")?.takeIf { it.isNotBlank() }?.let {
            File(it).absoluteFile.apply { parentFile?.mkdirs() }.writeText(summary, Charsets.UTF_8)
        }
    }

    // ------------------------------------------------------------- fixtures

    private fun writeTemp(name: String, bytes: ByteArray): File =
        File.createTempFile("live-replay-$name", ".wav").apply {
            deleteOnExit()
            writeBytes(bytes)
        }

    /** Minimal canonical WAV writer, used only to exercise [WavReader]. */
    private fun wav(samples: FloatArray, rate: Int, channels: Int, float: Boolean): ByteArray {
        val bytesPerSample = if (float) 4 else 2
        val data = ByteBuffer.allocate(samples.size * bytesPerSample).order(ByteOrder.LITTLE_ENDIAN)
        for (sample in samples) {
            if (float) data.putFloat(sample) else data.putShort(Math.round(sample * 32767f).toShort())
        }
        val payload = data.array()
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(36 + payload.size)
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16)
        header.putShort(if (float) 3 else 1)
        header.putShort(channels.toShort())
        header.putInt(rate)
        header.putInt(rate * channels * bytesPerSample)
        header.putShort((channels * bytesPerSample).toShort())
        header.putShort((bytesPerSample * 8).toShort())
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(payload.size)
        return ByteArrayOutputStream().apply {
            write(header.array())
            write(payload)
        }.toByteArray()
    }
}
