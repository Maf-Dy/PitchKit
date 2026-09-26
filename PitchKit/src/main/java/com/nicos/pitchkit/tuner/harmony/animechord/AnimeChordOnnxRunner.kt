package com.nicos.pitchkit.tuner.harmony.animechord

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** The heads of one window, already cut down to the frames that window commits. */
internal class AnimeChordHeads(
    val frames: Int,
    /** `frames x 745` row-major logits. */
    val rootChord: FloatArray,
    /** `frames x 13` row-major logits. */
    val bass: FloatArray,
    /** `frames x 13`; only read when the caller asks for the auxiliary heads. */
    val key: FloatArray? = null,
    val boundary: FloatArray? = null,
    val beat: FloatArray? = null,
    val downbeat: FloatArray? = null,
)

/**
 * ONNX Runtime wrapper for the opset-17 spectrogram-input export of
 * anime-song/Chord-Transcription.
 *
 * The graph's time axis is dynamic — the export declares `spec: [1, 2, frames, 252]` with
 * `frames` symbolic — so a window is run at its real length and the short final window is
 * not padded. The backbone down- and up-samples time by eight, so the heads come back
 * with `ceil(frames / 8) * 8` rows; the extra rows are the reference's own
 * `_match_time_length` trim, made an identity in the export and done here instead.
 *
 * The session is configured the way the measurement in `anime-windowed-report.md` §7 was
 * run: four intra-op threads, one inter-op, sequential execution. **The arena is the
 * binding cost of this engine**, not the clock: 455 MB of native allocation at the shipped
 * 30 s window (220 MB at 15 s, 910 MB at 45 s), because ORT materialises the attention
 * matrices torch's memory-efficient SDPA never writes down. Keeping one session alive for
 * the whole song is deliberate — the arena is reused window to window instead of being
 * torn down and re-grown.
 */
internal class AnimeChordOnnxRunner(
    modelBytes: ByteArray,
    threads: Int = defaultThreads(),
) : AutoCloseable {
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private var inputBuffer: FloatBuffer = allocate(0)
    private var closed = false

    init {
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(threads.coerceIn(1, 8))
        options.setInterOpNumThreads(1)
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        // The arena keeps the attention workspace between windows instead of returning it
        // to the OS and re-growing it on the next one.
        options.setCPUArenaAllocator(true)
        // Every window but the last has the same shape, but the shape is symbolic in the
        // graph, so the planner cannot pre-plan it; saying so costs nothing and avoids a
        // wasted first-run planning pass.
        options.setMemoryPatternOptimization(false)
        session = environment.createSession(modelBytes, options)

        require(AnimeChordContract.INPUT_NAME in session.inputNames) {
            "anime input '${AnimeChordContract.INPUT_NAME}' missing: ${session.inputNames}"
        }
        val expected = setOf(
            AnimeChordContract.ROOT_CHORD_OUTPUT,
            AnimeChordContract.BASS_OUTPUT,
            AnimeChordContract.KEY_OUTPUT,
            AnimeChordContract.BOUNDARY_OUTPUT,
            AnimeChordContract.BEAT_OUTPUT,
            AnimeChordContract.DOWNBEAT_OUTPUT,
        )
        require(session.outputNames.containsAll(expected)) {
            "Unexpected anime outputs: ${session.outputNames}"
        }
    }

    /**
     * @param features one channel of the standardised feature matrix, `frames x 252`. The
     * app decodes mono, so the stereo channel the graph wants is this one duplicated.
     * @param commitStart first window-local frame to return.
     * @param commitEnd one past the last window-local frame to return.
     */
    @Synchronized
    fun infer(
        features: FloatArray,
        frames: Int,
        commitStart: Int,
        commitEnd: Int,
        includeAuxiliary: Boolean = false,
    ): AnimeChordHeads {
        check(!closed) { "anime OrtSession is closed" }
        val bins = AnimeChordContract.INPUT_BINS
        require(frames > 0) { "a window must carry at least one frame" }
        require(features.size >= frames * bins) {
            "anime expects ${frames * bins} values, got ${features.size}"
        }
        require(commitStart in 0..commitEnd && commitEnd <= frames) {
            "commit range $commitStart..$commitEnd outside 0..$frames"
        }

        val values = frames * bins
        if (inputBuffer.capacity() < 2 * values) inputBuffer = allocate(2 * values)
        inputBuffer.clear()
        inputBuffer.put(features, 0, values)
        inputBuffer.put(features, 0, values)
        inputBuffer.flip()

        OnnxTensor.createTensor(
            environment,
            inputBuffer,
            longArrayOf(1L, 2L, frames.toLong(), bins.toLong()),
        ).use { input ->
            session.run(mapOf(AnimeChordContract.INPUT_NAME to input)).use { result ->
                val committed = commitEnd - commitStart
                return AnimeChordHeads(
                    frames = committed,
                    rootChord = slice(
                        result, AnimeChordContract.ROOT_CHORD_OUTPUT, frames,
                        AnimeChordContract.ROOT_CHORD_COUNT, commitStart, commitEnd,
                    ),
                    bass = slice(
                        result, AnimeChordContract.BASS_OUTPUT, frames,
                        AnimeChordContract.BASS_COUNT, commitStart, commitEnd,
                    ),
                    key = if (!includeAuxiliary) null else slice(
                        result, AnimeChordContract.KEY_OUTPUT, frames,
                        AnimeChordContract.KEY_COUNT, commitStart, commitEnd,
                    ),
                    boundary = if (!includeAuxiliary) null else slice(
                        result, AnimeChordContract.BOUNDARY_OUTPUT, frames, 1,
                        commitStart, commitEnd,
                    ),
                    beat = if (!includeAuxiliary) null else slice(
                        result, AnimeChordContract.BEAT_OUTPUT, frames, 1,
                        commitStart, commitEnd,
                    ),
                    downbeat = if (!includeAuxiliary) null else slice(
                        result, AnimeChordContract.DOWNBEAT_OUTPUT, frames, 1,
                        commitStart, commitEnd,
                    ),
                )
            }
        }
    }

    private fun slice(
        result: OrtSession.Result,
        name: String,
        frames: Int,
        width: Int,
        commitStart: Int,
        commitEnd: Int,
    ): FloatArray {
        val value = result.get(name)
            .orElseThrow { IllegalStateException("anime output '$name' missing") }
        val tensor = value as? OnnxTensor ?: error("anime output '$name' is not a tensor")
        val buffer = tensor.floatBuffer ?: error("anime output '$name' is not float32")
        val produced = buffer.remaining() / width
        require(buffer.remaining() == produced * width) {
            "anime output '$name' is not a multiple of $width"
        }
        // The backbone rounds its time axis up to a multiple of eight; the tail rows are
        // the trim the export turned into an identity.
        require(produced >= frames && produced - frames < AnimeChordContract.TIME_MULTIPLE) {
            "anime output '$name' produced $produced rows for a $frames-frame window"
        }
        val output = FloatArray((commitEnd - commitStart) * width)
        if (output.isEmpty()) return output
        val slice = buffer.duplicate()
        slice.position(slice.position() + commitStart * width)
        slice.get(output, 0, output.size)
        return output
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        session.close()
    }

    private fun allocate(values: Int): FloatBuffer = ByteBuffer
        .allocateDirect(values.coerceAtLeast(1) * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

    companion object {
        /**
         * Four threads is what the measurement assumed and what a current mid-range phone
         * has in big cores; more only contends for memory bandwidth on a graph this
         * memory-bound.
         */
        fun defaultThreads(): Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
    }
}
