package com.nicos.pitchkit.tuner.harmony.animechord

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeNoException
import org.junit.Test

/**
 * One real window through the shipped graph, compared against the forward pass the
 * reference Python runner saved, and the author's own HMM path over the result.
 *
 * The fixture is 648 frames (15.05 s) rather than the shipped 1292, because ONNX Runtime
 * materialises the attention matrices and a 30 s window costs ~455 MB of native arena
 * against ~220 MB for this one. The graph's time axis is symbolic, so the operators are
 * identical either way; the 30 s geometry is pinned separately in
 * `AnimeChordWindowPlanTest`.
 */
class AnimeChordOnnxParityTest {
    private val provenance = JSONObject(
        resource("animechord-window.json").toString(Charsets.UTF_8)
    )

    @Test
    fun theShippedGraphReproducesTheReferenceForwardPass() {
        assertEquals(1, provenance.getInt("schema_version"))
        assertEquals(
            AnimeChordContract.MODEL_SHA256,
            provenance.getString("onnx_sha256"),
        )
        val frames = provenance.getInt("frames")
        assertEquals(AnimeChordContract.ROOT_CHORD_COUNT, provenance.getInt("root_classes"))

        val spec = floats(resource("animechord-window-spec.f32"))
        assertEquals(frames * AnimeChordContract.INPUT_BINS, spec.size)
        val expectedRoot = floats(resource("animechord-window-root.f32"))
        assertEquals(frames * AnimeChordContract.ROOT_CHORD_COUNT, expectedRoot.size)
        val expectedOthers = floats(resource("animechord-window-heads.f32"))
        val otherWidth = AnimeChordContract.BASS_COUNT + AnimeChordContract.KEY_COUNT + 3
        assertEquals(frames * otherWidth, expectedOthers.size)

        // Android Studio's JBR bundles an MSVC runtime too old for onnxruntime.dll, so
        // the native library fails its DllMain there. That is a property of the JVM, not
        // of this port: the same convention as `LiveReplayTest.onnxRuntimeLoadsOnTheJvm`
        // applies, and the graph parity is measured on the patched JDK
        // `tools/accuracy_audit/live_replay.py --prepare-jvm` builds.
        val runner = try {
            AnimeChordOnnxRunner(
                AnimeChordTestAssets.read(AnimeChordContract.MODEL_FILE),
                threads = 2,
            )
        } catch (error: Throwable) {
            assumeNoException(
                "ONNX Runtime has no usable native library on this JVM " +
                    "(${System.getProperty("java.home")}); run this parity check on an " +
                    "ONNX-capable JDK - see live_replay.py --prepare-jvm",
                error,
            )
            return
        }
        val heads = runner.use { session ->
            session.infer(
                features = spec,
                frames = frames,
                commitStart = 0,
                commitEnd = frames,
                includeAuxiliary = true,
            )
        }
        assertEquals(frames, heads.frames)

        assertTrue("root_chord", compare("root_chord", heads.rootChord, expectedRoot) <= 1e-4)
        assertTrue(
            "bass",
            compare("bass", heads.bass, column(expectedOthers, frames, otherWidth, 0,
                AnimeChordContract.BASS_COUNT)) <= 1e-4,
        )
        assertTrue(
            "key",
            compare("key", requireNotNull(heads.key),
                column(expectedOthers, frames, otherWidth, AnimeChordContract.BASS_COUNT,
                    AnimeChordContract.KEY_COUNT)) <= 1e-4,
        )
        val auxiliaryBase = AnimeChordContract.BASS_COUNT + AnimeChordContract.KEY_COUNT
        assertTrue(
            "boundary",
            compare("boundary", requireNotNull(heads.boundary),
                column(expectedOthers, frames, otherWidth, auxiliaryBase, 1)) <= 1e-4,
        )
        assertTrue(
            "beat",
            compare("beat", requireNotNull(heads.beat),
                column(expectedOthers, frames, otherWidth, auxiliaryBase + 1, 1)) <= 1e-4,
        )
        assertTrue(
            "downbeat",
            compare("downbeat", requireNotNull(heads.downbeat),
                column(expectedOthers, frames, otherWidth, auxiliaryBase + 2, 1)) <= 1e-4,
        )
    }

    @Test
    fun theStickyViterbiWalksTheSamePathTheReferenceDid() {
        val frames = provenance.getInt("frames")
        val logits = floats(resource("animechord-window-root.f32"))
        val expected = intArray(provenance.getJSONArray("root_path"))
        assertEquals(frames, expected.size)

        val viterbi = AnimeChordViterbi(
            classCount = AnimeChordContract.ROOT_CHORD_COUNT,
            expectedFrames = frames,
        )
        for (frame in 0 until frames) {
            viterbi.add(logits, frame * AnimeChordContract.ROOT_CHORD_COUNT)
        }
        assertArrayEqualsWithReport("root", expected, viterbi.decode())

        val bassLogits = column(
            floats(resource("animechord-window-heads.f32")),
            frames,
            AnimeChordContract.BASS_COUNT + AnimeChordContract.KEY_COUNT + 3,
            0,
            AnimeChordContract.BASS_COUNT,
        )
        val bass = AnimeChordViterbi(
            classCount = AnimeChordContract.BASS_COUNT,
            expectedFrames = frames,
        )
        for (frame in 0 until frames) bass.add(bassLogits, frame * AnimeChordContract.BASS_COUNT)
        assertArrayEqualsWithReport(
            "bass",
            intArray(provenance.getJSONArray("bass_path")),
            bass.decode(),
        )
    }

    private fun assertArrayEqualsWithReport(tag: String, expected: IntArray, actual: IntArray) {
        assertEquals("$tag frame count", expected.size, actual.size)
        var mismatches = 0
        var first = -1
        for (index in expected.indices) {
            if (expected[index] != actual[index]) {
                if (first < 0) first = index
                mismatches++
            }
        }
        assertEquals(
            "$tag path: $mismatches/${expected.size} frames differ, first at $first " +
                (if (first >= 0) "(${expected[first]} vs ${actual[first]})" else ""),
            0,
            mismatches,
        )
    }

    private fun compare(tag: String, actual: FloatArray, expected: FloatArray): Double {
        assertEquals("$tag size", expected.size, actual.size)
        var scale = 0.0
        for (value in expected) scale = max(scale, abs(value.toDouble()))
        var worst = 0.0
        for (index in expected.indices) {
            worst = max(worst, abs(actual[index].toDouble() - expected[index].toDouble()))
        }
        println("animechord $tag: scale=$scale max|delta|=$worst relative=${worst / scale}")
        return worst / scale
    }

    private fun column(
        values: FloatArray,
        frames: Int,
        stride: Int,
        offset: Int,
        width: Int,
    ): FloatArray = FloatArray(frames * width) { index ->
        values[(index / width) * stride + offset + index % width]
    }

    private fun intArray(array: org.json.JSONArray): IntArray =
        IntArray(array.length()) { array.getInt(it) }

    private fun resource(name: String) =
        requireNotNull(javaClass.getResourceAsStream("/$name")) { name }.use { it.readBytes() }

    private fun floats(bytes: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }
}
