package com.nicos.pitchkit.tuner.live

import com.nicos.pitchkit.tuner.models.AudioFrame
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.File
import java.io.FileDescriptor
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.io.PrintStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The live replay's engine, driven from a pipe instead of a WAV file.
 *
 * [LiveReplay] answers *what would the phone have shown me on this recording*.
 * This answers the same question about a microphone that is open **right now**:
 * it is the identical construction — the shipped `ChordNetStreamingRecognizer`,
 * `CremaStreamingRecognizer`, `BtcStreamingRecognizer` or `PitchAnalyzer`, built
 * from the very same `src/main/assets` bytes, fed `AudioFrame`s at the engine's
 * own negotiated rate and buffer size — with stdin as the audio source and
 * stdout as the trace.
 *
 * Nothing in `src/main` changes, and the two deliberate differences from a
 * device that [LiveReplay] declares still hold here: there is no `TunerEngine`
 * RMS gate and no `Channel.CONFLATED` drop, and the recognizer's internal
 * decision fields (`raw_model`, `dsp_label`, `gesture_ready`) are only written
 * to `android.util.Log`, so they are emitted as `null` exactly as the replay
 * trace emits them. `backend` still separates the DSP rescue from the model.
 *
 * ## The wire protocol
 *
 * Both directions are little-endian and line-oriented respectively, chosen so a
 * Python parent can speak it with `struct` and `json` and nothing else.
 *
 * **stdin** — a stream of frames, each one
 *
 * ```
 *   int32  count     number of float32 samples that follow; < 0 means "stop"
 *   float32[count]   PCM, mono, already at the engine's own sample rate, -1..1
 * ```
 *
 * The parent does not have to respect the engine's buffer size: samples are
 * re-chunked here into exactly `engine.bufferSize`-sample `AudioFrame`s, which
 * is the geometry `GuitarTunerListener` asks `AudioCapture` for, so the
 * recognizer sees the device's cadence however the pipe happened to be written.
 *
 * **stdout** — one compact JSON object per line, UTF-8:
 *
 * ```
 *   {"type":"ready","engine":…,"sample_rate":…,"buffer_size":…,"startup_ms":…}
 *   {"type":"row","t_samples":…,"emitted":…,"confidence":…,"backend":…,
 *    "raw_model":null,"dsp_label":null,"gesture_ready":null}
 *   {"type":"end","t_samples":…,"frames":…,"emissions":…}
 *   {"type":"error","message":…}
 * ```
 *
 * A `row` is written **only when the recognizer emitted something** — that is
 * what "one JSON line per emitted result" means, and it keeps the pipe quiet
 * while the engine is still filling its context. `t_samples` is the index of
 * the last sample fed, the same audio-sample clock every `LiveReplay` trace
 * carries, so a row can be tied back to the exact chunk that produced it and
 * end-to-end latency measured against it.
 *
 * stdout is reserved for the protocol: [main] rebinds `System.out` to stderr
 * before anything else runs, because ONNX Runtime and the JVM itself are free
 * to print there and one stray line would desynchronise the parent's parser.
 *
 * Run it with a plain classpath, no Gradle, under the ONNX-capable JDK that
 * `tools/accuracy_audit/live_replay.py --prepare-jvm` builds:
 *
 * ```
 *   .accuracy-work/tools/jbr-ort/bin/java.exe -cp @classpath \
 *       com.nicos.pitchkit.tuner.live.LiveEngineServerKt crema
 * ```
 *
 * `tools/accuracy_audit/live_engine_classpath.py` caches that classpath, so the
 * per-request cost is a JVM start and one model load rather than a Gradle
 * invocation.
 */
object LiveEngineServer {

    /** Stop sentinel in the `count` header. Any negative value stops the loop. */
    const val STOP: Int = -1

    /**
     * Walk up from the working directory to the module's `src/main/assets`, the
     * same search `LiveReplayTest` does, so the server can be started from the
     * repository root or from the module directory.
     */
    fun defaultAssets(): File? {
        var cursor: File? = File(System.getProperty("user.dir") ?: ".").absoluteFile
        while (cursor != null) {
            for (prefix in listOf("src/main/assets", "PitchKit/PitchKit/src/main/assets")) {
                val candidate = File(cursor, prefix)
                if (candidate.isDirectory) return candidate
            }
            cursor = cursor.parentFile
        }
        return null
    }

    /**
     * Build [engineId] from [assets] and pump [input] into it until the stream
     * ends or a negative count arrives.
     *
     * @return the number of emitted rows written to [output].
     */
    fun run(engineId: String, assets: File, input: InputStream, output: OutputStream,
            log: PrintStream): Int {
        val engine = LiveReplay.Engine.of(engineId)
        val writer = BufferedOutputStream(output)

        val startedAt = System.nanoTime()
        val runner = try {
            LiveReplay.buildEngine(engine, assets, JSONObject())
        } catch (error: Throwable) {
            writeLine(writer, JSONObject().apply {
                put("type", "error")
                put("message", "${error.javaClass.simpleName}: ${error.message}")
            })
            writer.flush()
            throw error
        }
        val startupMs = (System.nanoTime() - startedAt) / 1e6

        writeLine(writer, JSONObject().apply {
            put("type", "ready")
            put("engine", engine.id)
            put("sample_rate", engine.sampleRate)
            put("buffer_size", engine.bufferSize)
            put("frame_period_s", engine.framePeriodSeconds)
            put("dsp_rescue", engine.dspRescue)
            put("startup_ms", Math.round(startupMs * 1000.0) / 1000.0)
            put("java_version", System.getProperty("java.version"))
        })
        writer.flush()
        log.println("live-engine: ${engine.id} ready in ${"%.0f".format(startupMs)} ms " +
            "(${engine.sampleRate} Hz, ${engine.bufferSize}-sample buffers)")

        val stream = DataInputStream(input.buffered(1 shl 16))
        // The recognizer wants whole buffers of the device's own size; the pipe
        // carries whatever the parent had. Everything short of a full buffer waits
        // here rather than being fed as a runt frame the phone would never produce.
        val pending = FloatArray(engine.bufferSize)
        var pendingCount = 0
        var fed = 0L
        var frames = 0
        var emissions = 0

        try {
            loop@ while (true) {
                val count = readInt(stream) ?: break@loop
                if (count < 0) break@loop
                if (count == 0) continue@loop
                var remaining = count
                val scratch = ByteArray(minOf(remaining, 1 shl 16) * 4)
                while (remaining > 0) {
                    val take = minOf(remaining, scratch.size / 4)
                    stream.readFully(scratch, 0, take * 4)
                    val buffer = ByteBuffer.wrap(scratch, 0, take * 4).order(ByteOrder.LITTLE_ENDIAN)
                    for (index in 0 until take) {
                        pending[pendingCount++] = buffer.float
                        if (pendingCount == engine.bufferSize) {
                            fed += pendingCount
                            frames++
                            val recognition = runner.feed(
                                AudioFrame(pending.copyOf(), engine.sampleRate, 1)
                            )
                            pendingCount = 0
                            if (recognition != null) {
                                emissions++
                                writeLine(writer, JSONObject().apply {
                                    put("type", "row")
                                    put("t_samples", fed)
                                    put("emitted", recognition.label)
                                    put("confidence", round6(recognition.confidence.toDouble()))
                                    put("backend", recognition.backend ?: JSONObject.NULL)
                                    // Not observable without a production change; see
                                    // LiveReplay's class documentation.
                                    put("raw_model", JSONObject.NULL)
                                    put("dsp_label", JSONObject.NULL)
                                    put("gesture_ready", JSONObject.NULL)
                                })
                                writer.flush()
                            }
                        }
                    }
                    remaining -= take
                }
            }
        } finally {
            runner.close()
            writeLine(writer, JSONObject().apply {
                put("type", "end")
                put("t_samples", fed)
                put("frames", frames)
                put("emissions", emissions)
            })
            writer.flush()
        }
        return emissions
    }

    private fun round6(value: Double): Double = Math.round(value * 1e6) / 1e6

    private fun writeLine(output: OutputStream, json: JSONObject) {
        output.write(json.toString().toByteArray(Charsets.UTF_8))
        output.write('\n'.code)
    }

    /** Little-endian int32, or null at a clean end of stream. */
    private fun readInt(stream: DataInputStream): Int? {
        val bytes = ByteArray(4)
        var read = 0
        while (read < 4) {
            val got = stream.read(bytes, read, 4 - read)
            if (got < 0) return null
            read += got
        }
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).int
    }
}

/**
 * `java -cp … com.nicos.pitchkit.tuner.live.LiveEngineServerKt <engine> [assetsDir]`.
 *
 * `<engine>` is any [LiveReplay.Engine] id — `chordnet`, `crema`, `btc`,
 * `classic`, and the `-norescue` / `-rescue` / `-rescue60` ablation variants.
 */
fun main(args: Array<String>) {
    // stdout belongs to the protocol from here on. Hold the real descriptor and
    // send everything the JVM, ONNX Runtime or the recognizers print to stderr.
    val protocol = FileOutputStream(FileDescriptor.out)
    val log = PrintStream(FileOutputStream(FileDescriptor.err), true)
    System.setOut(log)

    if (args.isEmpty()) {
        log.println("usage: LiveEngineServerKt <engine> [assetsDir]")
        log.println("engines: " + LiveReplay.Engine.entries.joinToString(", ") { it.id })
        System.exit(2)
        return
    }
    val assets = if (args.size > 1) File(args[1]).absoluteFile else LiveEngineServer.defaultAssets()
    if (assets == null || !assets.isDirectory) {
        protocol.write(
            (JSONObject().apply {
                put("type", "error")
                put("message", "assets directory not found (pass it as the second argument)")
            }.toString() + "\n").toByteArray(Charsets.UTF_8)
        )
        protocol.flush()
        System.exit(3)
        return
    }
    try {
        LiveEngineServer.run(args[0], assets, System.`in`, protocol, log)
    } catch (error: Throwable) {
        log.println("live-engine: failed - ${error.javaClass.name}: ${error.message}")
        error.printStackTrace(log)
        System.exit(1)
        return
    }
    System.exit(0)
}
