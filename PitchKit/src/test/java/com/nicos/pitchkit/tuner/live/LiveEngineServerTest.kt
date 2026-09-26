package com.nicos.pitchkit.tuner.live

import ai.onnxruntime.OrtEnvironment
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetContract
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeNoException
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.PrintStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * [LiveEngineServer] driven through its own pipe, in-process.
 *
 * The microphone panel on the Live page is only as trustworthy as this seam: the
 * browser's audio becomes length-prefixed float32 on the engine's stdin, and what
 * the page draws is whatever JSON came back on stdout. So the test writes the
 * protocol by hand — including a deliberately awkward chunking that never lines up
 * with the engine's buffer size — and reads every line back as JSON.
 *
 * Classic DSP carries the load-bearing cases because it needs no ONNX Runtime and
 * no downloaded asset, so this runs everywhere `testDebugUnitTest` runs. The
 * ChordNet case repeats the same assertions against a real model and assumes out
 * when the assets or the native library are missing, exactly as the rest of the
 * live suite does.
 */
class LiveEngineServerTest {

    private val log = PrintStream(ByteArrayOutputStream())

    @Test
    fun classicEmitsJsonLinesForTwoSecondsOfAudio() {
        val assets = assetsDir()
        val engine = LiveReplay.Engine.CLASSIC
        val audio = triad(seconds = 2.0, rate = engine.sampleRate)
        // 1500 samples is coprime with nothing in particular and never equals the
        // 8192-sample buffer: the server has to re-chunk, which is what the browser
        // (whose chunks are ~186 ms at its own rate) will actually make it do.
        val lines = drive(engine.id, assets, audio, chunk = 1500)

        val ready = lines.first()
        assertEquals("ready", ready.getString("type"))
        assertEquals(engine.id, ready.getString("engine"))
        assertEquals(engine.sampleRate, ready.getInt("sample_rate"))
        assertEquals(engine.bufferSize, ready.getInt("buffer_size"))
        assertTrue("startup_ms must be reported", ready.has("startup_ms"))
        assertTrue(
            "startup must be a real measurement, not a placeholder",
            ready.getDouble("startup_ms") >= 0.0,
        )

        val end = lines.last()
        assertEquals("end", end.getString("type"))
        // Only whole buffers are fed; the runt at the tail waits for audio that never
        // came, exactly as it would between two microphone chunks.
        val whole = (audio.size / engine.bufferSize) * engine.bufferSize
        assertEquals(whole.toLong(), end.getLong("t_samples"))
        assertEquals(audio.size / engine.bufferSize, end.getInt("frames"))

        val rows = lines.filter { it.getString("type") == "row" }
        assertEquals(
            "the end row must count the rows that were written",
            rows.size,
            end.getInt("emissions"),
        )
        assertTrue("Classic DSP emitted nothing for two seconds of a C major triad",
            rows.isNotEmpty())
        var previous = 0L
        for (row in rows) {
            // The trace contract the page and live_metrics both read.
            for (key in listOf("t_samples", "emitted", "confidence", "backend",
                    "raw_model", "dsp_label", "gesture_ready")) {
                assertTrue("row is missing $key: $row", row.has(key))
            }
            val samples = row.getLong("t_samples")
            assertTrue("t_samples must not go backwards", samples >= previous)
            assertTrue("t_samples must be a whole number of buffers",
                samples % engine.bufferSize == 0L)
            assertTrue("t_samples past the audio fed", samples <= whole)
            previous = samples
            assertNotNull(row.getString("emitted"))
            // The fields the recognizer only writes to android.util.Log stay null here,
            // exactly as LiveReplay's trace declares.
            assertTrue(row.isNull("raw_model"))
            assertTrue(row.isNull("dsp_label"))
            assertTrue(row.isNull("gesture_ready"))
        }
    }

    /** A negative count is the protocol's stop, and nothing after it is read. */
    @Test
    fun aNegativeCountStopsTheEngineAndTheRestOfThePipeIsIgnored() {
        val assets = assetsDir()
        val engine = LiveReplay.Engine.CLASSIC
        val audio = triad(seconds = 1.0, rate = engine.sampleRate)
        val body = ByteArrayOutputStream()
        body.write(frame(audio))
        body.write(int32(LiveEngineServer.STOP))
        body.write(frame(triad(seconds = 5.0, rate = engine.sampleRate)))  // never read

        val output = ByteArrayOutputStream()
        LiveEngineServer.run(engine.id, assets, ByteArrayInputStream(body.toByteArray()),
            output, log)
        val lines = parse(output)
        val whole = (audio.size / engine.bufferSize) * engine.bufferSize
        assertEquals(whole.toLong(), lines.last().getLong("t_samples"))
    }

    /** An empty pipe is a valid session: ready, then end, and no rows in between. */
    @Test
    fun anEmptySessionIsReadyThenEnd() {
        val output = ByteArrayOutputStream()
        LiveEngineServer.run("classic", assetsDir(), ByteArrayInputStream(ByteArray(0)),
            output, log)
        val lines = parse(output)
        assertEquals(2, lines.size)
        assertEquals("ready", lines[0].getString("type"))
        assertEquals("end", lines[1].getString("type"))
        assertEquals(0L, lines[1].getLong("t_samples"))
    }

    /**
     * The same protocol over the shipped ChordNet model, so the microphone panel's
     * phone lanes are covered by more than the Classic DSP stand-in.
     */
    @Test
    fun chordNetStreamsThroughTheSamePipe() {
        val assets = assetsDir()
        assumeTrue(
            "ChordNet assets are not installed",
            File(assets, "chordnet/${ChordNetContract.MODEL_FILE}").isFile,
        )
        try {
            OrtEnvironment.getEnvironment()
        } catch (error: Throwable) {
            assumeNoException("ONNX Runtime has no usable native library on this JVM", error)
            return
        }
        val engine = LiveReplay.Engine.CHORD_NET
        val audio = triad(seconds = 3.0, rate = engine.sampleRate)
        val lines = drive(engine.id, assets, audio, chunk = 1024)
        val ready = lines.first()
        assertEquals(ChordNetContract.SAMPLE_RATE, ready.getInt("sample_rate"))
        assertEquals(ChordNetContract.HOP_LENGTH, ready.getInt("buffer_size"))
        assertTrue("the rescue lane state must be declared", ready.has("dsp_rescue"))
        val rows = lines.filter { it.getString("type") == "row" }
        assertTrue("ChordNet emitted nothing for three seconds of a triad", rows.isNotEmpty())
        for (row in rows) {
            val backend = row.optString("backend")
            assertTrue(
                "Unexpected ChordNet backend: $backend",
                backend == "ChordNet 2E1D" || backend == "ChordNet + temporal CQT DSP",
            )
        }
    }

    // ------------------------------------------------------------- fixtures

    private fun assetsDir(): File {
        val assets = LiveEngineServer.defaultAssets()
        assertNotNull("src/main/assets was not found from ${System.getProperty("user.dir")}",
            assets)
        return assets!!
    }

    /** Write `audio` as `chunk`-sample frames, then STOP, and collect the JSON back. */
    private fun drive(engine: String, assets: File, audio: FloatArray,
                      chunk: Int): List<JSONObject> {
        val body = ByteArrayOutputStream()
        var offset = 0
        while (offset < audio.size) {
            val size = minOf(chunk, audio.size - offset)
            body.write(frame(audio.copyOfRange(offset, offset + size)))
            offset += size
        }
        body.write(int32(LiveEngineServer.STOP))
        val output = ByteArrayOutputStream()
        LiveEngineServer.run(engine, assets, ByteArrayInputStream(body.toByteArray()),
            output, log)
        val lines = parse(output)
        assertTrue("the engine wrote nothing at all", lines.size >= 2)
        return lines
    }

    private fun parse(output: ByteArrayOutputStream): List<JSONObject> =
        output.toString(Charsets.UTF_8.name())
            .split('\n')
            .filter { it.isNotBlank() }
            .map { JSONObject(it) }

    private fun int32(value: Int): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()

    private fun frame(samples: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(4 + samples.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(samples.size)
        for (sample in samples) buffer.putFloat(sample)
        return buffer.array()
    }

    /** The same deterministic C major triad `LiveReplayTest` feeds its recognizers. */
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
            val envelope = Math.min(1.0, t / 0.08)
            samples[index] = (value * 0.12 * envelope).toFloat()
        }
        return samples
    }
}
