package com.nicos.pitchkit.tuner.live

import com.nicos.pitchkit.tuner.DetectionMode
import com.nicos.pitchkit.tuner.PitchAnalyzer
import com.nicos.pitchkit.tuner.TuningResult
import com.nicos.pitchkit.tuner.harmony.ChordRecognition
import com.nicos.pitchkit.tuner.harmony.ChordRecognizer
import com.nicos.pitchkit.tuner.harmony.btc.BtcContract
import com.nicos.pitchkit.tuner.harmony.btc.BtcMetadata
import com.nicos.pitchkit.tuner.harmony.btc.BtcStreamingRecognizer
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetContract
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetStreamingRecognizer
import com.nicos.pitchkit.tuner.harmony.crema.CremaContract
import com.nicos.pitchkit.tuner.harmony.crema.CremaStreamingRecognizer
import com.nicos.pitchkit.tuner.harmony.solitito.SolititoContract
import com.nicos.pitchkit.tuner.harmony.solitito.SolititoStreamingRecognizer
import com.nicos.pitchkit.tuner.models.AudioFrame
import com.nicos.pitchkit.tuner.models.InstrumentProfile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * JVM replay of the *live* chord path — Level 0 of the live benchmark plan.
 *
 * Nothing in `src/main` changes. The shipped streaming recognizers already take
 * model bytes rather than a `Context`, so this harness loads the very same
 * assets off disk, builds the very same recognizer, and feeds it `AudioFrame`s
 * at the engine's own rate and buffer size — the geometry `GuitarTunerListener`
 * asks `AudioCapture` for.
 *
 * Two deliberate differences from the device, both stated in the report:
 *
 *  * **No RMS gate and no conflation.** `TunerEngine`'s gate and its
 *    `Channel.CONFLATED` transport are Android-side; every frame is delivered
 *    here, deterministically, exactly once. That is the recommended split —
 *    deterministic on the JVM, real drops measured on device.
 *  * **No internal decision fields.** `raw_model`, `model_conf`, `dsp_label`,
 *    `dsp_score`, `gesture_ready` and `gesture_updates` live inside the
 *    recognizer and are only written to `android.util.Log`, which is a no-op
 *    under unit tests. Surfacing them would be a production change, so they are
 *    emitted as `null`. `backend` still distinguishes the DSP rescue from the
 *    model ("ChordNet + temporal CQT DSP" vs "ChordNet 2E1D"), which is what the
 *    extended-label-flicker attribution actually needs.
 *
 * Every row is timestamped by the **sample index of the last sample fed**, never
 * by wall clock, so traces and `.lab` ground truth share one exact clock.
 */
object LiveReplay {

    enum class Engine(
        val id: String,
        /** The rate `GuitarTunerListener` asks `AudioCapture` to negotiate. */
        val sampleRate: Int,
        /** The `bufferSize` (a sample count, not a duration) for that engine. */
        val bufferSize: Int,
        /**
         * Whether the shared `CqtChordTemplateDetector` rescue lane is live.
         * Shipped lanes now carry `false` — the rescue ablation turned it off by
         * default on all three model lanes. The `-rescue` / `-rescue60` variants
         * carry `true` so the pre-ablation behaviour stays benchmarkable, and the
         * `-norescue` variants pin `false` explicitly so a scored directory keeps
         * its meaning whatever the defaults do next. Classic DSP has no rescue
         * lane at all and is `false`.
         */
        val dspRescue: Boolean = false,
        /**
         * The model-confidence ceiling the rescue is gated by when it is on at
         * all. [DEFAULT_RESCUE_THRESHOLD] is the current
         * `DSP_OVERRIDE_MODEL_CONFIDENCE` on all three model lanes;
         * [LEGACY_RESCUE_THRESHOLD] is what shipped before the ablation.
         */
        val rescueConfidenceThreshold: Double = DEFAULT_RESCUE_THRESHOLD,
        /**
         * solitito's own noise gate, in dBFS on the raw frame RMS. `null` for every lane
         * that does not have one. The shipped port carries
         * `SolititoContract.GATE_DB`; the `-gate34` variant carries upstream's own
         * constant so the ablation the Stage C screening measured can be reproduced by
         * the JVM harness rather than only in Python.
         */
        val gateDb: Double? = null,
        /**
         * Whether to feed the highest-rate rendering when the corpus has no audio at this
         * engine's own rate. Only solitito sets it: it resamples everything itself, so
         * "least pre-damaged" is the right choice and it is also what the Python
         * reference's `pick_audio` does, which the parity story depends on. Every other
         * lane matches its rate exactly on both corpora and never reaches this.
         */
        val preferHighestRate: Boolean = false,
    ) {
        CHORD_NET("chordnet", ChordNetContract.SAMPLE_RATE, ChordNetContract.HOP_LENGTH),
        CREMA("crema", CremaContract.SAMPLE_RATE, CremaContract.HOP_LENGTH),
        // The phone's fourth live lane. `GuitarTunerListener` gives a
        // BtcStreamingRecognizer the same geometry as ChordNet - 22 050 Hz and a
        // 2 048-sample buffer - so the replay does too.
        BTC("btc", BtcContract.SAMPLE_RATE, BtcContract.HOP_LENGTH),
        // The solitito-ai lane: 16 kHz, one 256-sample hop per buffer, no rescue lane.
        SOLITITO(
            "solitito", SolititoContract.SAMPLE_RATE, SolititoContract.HOP_LENGTH,
            dspRescue = false, gateDb = SolititoContract.GATE_DB, preferHighestRate = true,
        ),
        // Classic DSP is the rescue detector's cousin, not a client of it.
        CLASSIC("classic", 44_100, 8_192, dspRescue = false),

        // ------------------------------------------- rescue ablation variants
        // Same assets, same geometry, same construction as the lane above them;
        // the only difference is the `dspRescue` / `rescueConfidenceThreshold`
        // arguments handed to the shipped recognizer. These exist so the rescue
        // can be measured, not shipped: nothing selects them on a device.
        //
        // The `-norescue` ids now describe what the shipped lanes do, and are
        // kept so the ablation report's scored directories keep resolving and so
        // the configuration stays pinned here rather than tracking a default.
        // The `-rescue` ids are the pre-ablation behaviour - rescue on at 0.80 -
        // so every row in that report remains reproducible from this harness.
        CHORD_NET_NO_RESCUE(
            "chordnet-norescue", ChordNetContract.SAMPLE_RATE, ChordNetContract.HOP_LENGTH,
            dspRescue = false,
        ),
        CHORD_NET_RESCUE(
            "chordnet-rescue", ChordNetContract.SAMPLE_RATE, ChordNetContract.HOP_LENGTH,
            dspRescue = true,
            rescueConfidenceThreshold = 0.80, // == LEGACY_RESCUE_THRESHOLD
        ),
        CHORD_NET_RESCUE_60(
            "chordnet-rescue60", ChordNetContract.SAMPLE_RATE, ChordNetContract.HOP_LENGTH,
            dspRescue = true,
            rescueConfidenceThreshold = 0.60,
        ),
        CREMA_NO_RESCUE(
            "crema-norescue", CremaContract.SAMPLE_RATE, CremaContract.HOP_LENGTH,
            dspRescue = false,
        ),
        CREMA_RESCUE(
            "crema-rescue", CremaContract.SAMPLE_RATE, CremaContract.HOP_LENGTH,
            dspRescue = true,
            rescueConfidenceThreshold = 0.80, // == LEGACY_RESCUE_THRESHOLD
        ),
        BTC_NO_RESCUE(
            "btc-norescue", BtcContract.SAMPLE_RATE, BtcContract.HOP_LENGTH,
            dspRescue = false,
        ),
        BTC_RESCUE(
            "btc-rescue", BtcContract.SAMPLE_RATE, BtcContract.HOP_LENGTH,
            dspRescue = true,
            rescueConfidenceThreshold = 0.80, // == LEGACY_RESCUE_THRESHOLD
        ),
        // solitito with upstream's own -34 dBFS gate instead of the port's -60. The one
        // constant the Stage C screening measured as worth +15.1 points on real guitar;
        // kept selectable so the port can prove it reproduces both sides of that result.
        SOLITITO_GATE34(
            "solitito-gate34", SolititoContract.SAMPLE_RATE, SolititoContract.HOP_LENGTH,
            dspRescue = false, gateDb = SolititoContract.SHIPPED_GATE_DB,
            preferHighestRate = true,
        ),
        ;

        val framePeriodSeconds: Double get() = bufferSize.toDouble() / sampleRate

        companion object {
            /** The current `DSP_OVERRIDE_MODEL_CONFIDENCE` on all three model lanes. */
            const val DEFAULT_RESCUE_THRESHOLD = 0.60

            /**
             * The gate that shipped before the rescue ablation, kept so the
             * `-rescue` variants can reproduce the report's baseline rows.
             */
            const val LEGACY_RESCUE_THRESHOLD = 0.80

            fun of(id: String): Engine = entries.firstOrNull { it.id == id.lowercase() }
                ?: error("Unknown live engine '$id'; expected one of ${entries.map(Engine::id)}")
        }
    }

    /** One emitted row of the JSONL trace. */
    data class Row(
        val index: Int,
        val audioSamples: Long,
        val audioSeconds: Double,
        val emitted: String?,
        val confidence: Double?,
        val backend: String?,
        val displayed: String?,
        val wallNanos: Long,
    )

    data class Item(val id: String, val audioByRate: Map<Int, File>, val lab: File?)

    data class Result(val engine: Engine, val item: Item, val trace: File, val meta: File, val rows: Int)

    // ---------------------------------------------------------------- driving

    /**
     * Run every (engine × item) pair in [manifestFile], writing one JSONL trace
     * and one meta sidecar per pair.
     *
     * Single-threaded and one file at a time on purpose: `recognize` is
     * `@Synchronized` and stateful, `OrtEnvironment` is process-wide, and the
     * machine is shared.
     */
    fun runManifest(manifestFile: File): List<Result> {
        val manifest = JSONObject(manifestFile.readText(Charsets.UTF_8))
        val assets = File(manifest.getString("assets_dir")).absoluteFile
        val outDir = File(manifest.getString("out_dir")).absoluteFile
        val engines = manifest.getJSONArray("engines").let { array ->
            (0 until array.length()).map { Engine.of(array.getString(it)) }
        }
        val items = manifest.getJSONArray("items").let { array ->
            (0 until array.length()).map { readItem(array.getJSONObject(it)) }
        }
        require(engines.isNotEmpty()) { "Manifest lists no engines" }
        require(items.isNotEmpty()) { "Manifest lists no items" }
        outDir.mkdirs()

        val results = mutableListOf<Result>()
        for (engine in engines) {
            val engineDir = File(outDir, engine.id).also { it.mkdirs() }
            for (item in items) {
                results += replayOne(engine, item, assets, engineDir)
                System.out.println("live-replay: ${engine.id}/${item.id} done")
            }
        }
        return results
    }

    private fun readItem(json: JSONObject): Item {
        val audio = mutableMapOf<Int, File>()
        val byRate = json.getJSONObject("audio")
        for (key in byRate.keys()) audio[key.toInt()] = File(byRate.getString(key)).absoluteFile
        require(audio.isNotEmpty()) { "Item ${json.getString("id")} has no audio" }
        val lab = if (json.has("lab")) File(json.getString("lab")).absoluteFile else null
        return Item(json.getString("id"), audio, lab)
    }

    private fun replayOne(engine: Engine, item: Item, assets: File, outDir: File): Result {
        // Prefer audio already rendered at the engine's own rate, so the
        // recognizer's StreamingPcmResampler is a passthrough and we measure the
        // engine, not the resampler.
        val matched = item.audioByRate[engine.sampleRate]
        val fallback = if (engine.preferHighestRate) {
            // Deterministic and highest-first; `audioByRate` is a HashMap, so iteration
            // order is not a thing to rely on.
            item.audioByRate.entries.maxByOrNull { it.key }!!.value
        } else {
            item.audioByRate.values.first()
        }
        val file = matched ?: fallback
        val audio = WavReader.read(file)
        val bufferFrames = frameChunk(engine, audio.sampleRate)

        val trace = File(outDir, "${item.id}.jsonl")
        val meta = File(outDir, "${item.id}.meta.json")
        val mono = audio.toMono()

        val provenance = JSONObject()
        val rows = mutableListOf<Row>()
        var displayed: String? = null
        var wallTotal = 0L

        buildEngine(engine, assets, provenance).use { runner ->
            var offset = 0
            var index = 0
            while (offset < mono.size) {
                val size = minOf(bufferFrames, mono.size - offset)
                val chunk = mono.copyOfRange(offset, offset + size)
                offset += size
                val startedAt = System.nanoTime()
                val recognition = runner.feed(AudioFrame(chunk, audio.sampleRate, 1))
                val wall = System.nanoTime() - startedAt
                wallTotal += wall
                if (recognition != null) displayed = recognition.label
                // The clock is the audio itself: the index of the last sample fed.
                rows += Row(
                    index = index++,
                    audioSamples = offset.toLong(),
                    audioSeconds = offset.toDouble() / audio.sampleRate,
                    emitted = recognition?.label,
                    confidence = recognition?.confidence,
                    backend = recognition?.backend,
                    displayed = displayed,
                    wallNanos = wall,
                )
            }
        }

        trace.bufferedWriter(Charsets.UTF_8).use { writer ->
            for (row in rows) {
                writer.write(rowJson(row).toString())
                writer.write("\n")
            }
        }

        provenance.put("schema_version", 1)
        provenance.put("engine", engine.id)
        provenance.put("item", item.id)
        provenance.put("engine_sample_rate", engine.sampleRate)
        provenance.put("engine_buffer_size", engine.bufferSize)
        provenance.put("engine_frame_period_s", engine.framePeriodSeconds)
        // The rescue ablation's two knobs, recorded on every trace so a scored
        // directory can never be mistaken for the shipped configuration.
        provenance.put("dsp_rescue", engine.dspRescue)
        provenance.put("gate_db", engine.gateDb ?: JSONObject.NULL)
        provenance.put(
            "rescue_confidence_threshold",
            if (engine.dspRescue) engine.rescueConfidenceThreshold else JSONObject.NULL,
        )
        provenance.put("audio_path", file.absolutePath)
        provenance.put("audio_sha256", sha256(file.readBytes()))
        provenance.put("audio_sample_rate", audio.sampleRate)
        provenance.put("audio_channels", audio.channelCount)
        provenance.put("audio_seconds", audio.durationSeconds)
        provenance.put("audio_rendered_at_engine_rate", matched != null)
        provenance.put("feed_chunk_samples", bufferFrames)
        provenance.put("rows", rows.size)
        provenance.put("wall_total_ns", wallTotal)
        provenance.put("realtime_factor", (wallTotal / 1e9) / audio.durationSeconds.coerceAtLeast(1e-9))
        provenance.put("instrument_profile", if (engine == Engine.CLASSIC) "Guitar" else JSONObject.NULL)
        provenance.put("lab", item.lab?.absolutePath ?: JSONObject.NULL)
        provenance.put("java_version", System.getProperty("java.version"))
        provenance.put(
            "note",
            "Deterministic JVM replay: every frame delivered, no TunerEngine RMS gate and no " +
                "Channel.CONFLATED drops. Internal decision fields are not observable without a " +
                "production change and are emitted as null.",
        )
        meta.writeText(provenance.toString(2), Charsets.UTF_8)
        return Result(engine, item, trace, meta, rows.size)
    }

    /**
     * How many source samples make one delivered buffer. When the audio is at the
     * engine's own rate this is exactly `bufferSize`; otherwise it is the same
     * duration at the file's rate, matching `AudioCapture`'s `bufferSize / rate`
     * cadence for a negotiated rate that is not the preferred one.
     */
    private fun frameChunk(engine: Engine, audioRate: Int): Int =
        Math.round(engine.bufferSize.toDouble() * audioRate / engine.sampleRate).toInt().coerceAtLeast(1)

    private fun rowJson(row: Row): JSONObject = JSONObject().apply {
        put("row", row.index)
        put("audio_samples", row.audioSamples)
        put("audio_s", round9(row.audioSeconds))
        put("emitted", row.emitted ?: JSONObject.NULL)
        put("confidence", row.confidence?.let(::round6) ?: JSONObject.NULL)
        put("backend", row.backend ?: JSONObject.NULL)
        put("displayed", row.displayed ?: JSONObject.NULL)
        // Internal recognizer state is not observable at Level 0; see the class doc.
        put("raw_model", JSONObject.NULL)
        put("model_conf", JSONObject.NULL)
        put("dsp_label", JSONObject.NULL)
        put("dsp_score", JSONObject.NULL)
        put("dsp_used", row.backend?.contains("DSP")?.let { it as Any } ?: JSONObject.NULL)
        put("gesture_ready", JSONObject.NULL)
        put("gesture_updates", JSONObject.NULL)
        put("wall_ns", row.wallNanos)
    }

    private fun round9(value: Double): Double = Math.round(value * 1e9) / 1e9
    private fun round6(value: Double): Double = Math.round(value * 1e6) / 1e6

    // --------------------------------------------------------------- engines

    /**
     * Uniform "feed one buffer, get the emitted recognition or null" seam.
     *
     * Public because [LiveEngineServer] drives the very same construction from a
     * pipe instead of a WAV file; both are test-source-set harnesses and neither
     * is visible to `src/main`.
     */
    interface EngineRunner : AutoCloseable {
        fun feed(frame: AudioFrame): ChordRecognition?
    }

    private class RecognizerRunner(private val recognizer: ChordRecognizer) : EngineRunner {
        override fun feed(frame: AudioFrame): ChordRecognition? = recognizer.recognize(frame)
        override fun close() = recognizer.close()
    }

    /** Classic DSP: `PitchAnalyzer` in CHORD mode, the live fallback lane. */
    private class ClassicRunner(private val analyzer: PitchAnalyzer) : EngineRunner {
        override fun feed(frame: AudioFrame): ChordRecognition? =
            when (val result = analyzer.process(frame)) {
                is TuningResult.Chord -> ChordRecognition(result.name, result.confidence, result.backend)
                else -> null
            }

        override fun close() = analyzer.reset()
    }

    fun buildEngine(engine: Engine, assets: File, provenance: JSONObject): EngineRunner =
        when (engine) {
            Engine.CHORD_NET, Engine.CHORD_NET_NO_RESCUE, Engine.CHORD_NET_RESCUE,
            Engine.CHORD_NET_RESCUE_60 -> {
                val model = File(assets, "chordnet/${ChordNetContract.MODEL_FILE}").readBytes()
                val plan = File(assets, "chordnet/${ChordNetContract.PLAN_FILE}").readBytes()
                provenance.put("model_sha256", JSONObject().apply {
                    put(ChordNetContract.MODEL_FILE, sha256(model))
                    put(ChordNetContract.PLAN_FILE, sha256(plan))
                })
                RecognizerRunner(
                    ChordNetStreamingRecognizer(
                        modelBytes = model,
                        planBytes = plan,
                        dspRescue = engine.dspRescue,
                        rescueConfidenceThreshold = engine.rescueConfidenceThreshold,
                    )
                )
            }

            Engine.CREMA, Engine.CREMA_NO_RESCUE, Engine.CREMA_RESCUE -> {
                val model = File(assets, "crema/${CremaContract.MODEL_FILE}").readBytes()
                val state = File(assets, "crema/${CremaContract.STATE_FILE}").readBytes()
                val planH1 = File(assets, "crema/crema-cqt-h1.bin").readBytes()
                val planH2 = File(assets, "crema/crema-cqt-h2.bin").readBytes()
                require(sha256(model) == CremaContract.MODEL_SHA256) { "Unexpected Crema model SHA-256" }
                require(sha256(state) == CremaContract.STATE_SHA256) { "Unexpected Crema state SHA-256" }
                require(sha256(planH1) == CremaContract.PLAN_H1_SHA256) { "Unexpected Crema h1 plan SHA-256" }
                require(sha256(planH2) == CremaContract.PLAN_H2_SHA256) { "Unexpected Crema h2 plan SHA-256" }
                provenance.put("model_sha256", JSONObject().apply {
                    put(CremaContract.MODEL_FILE, sha256(model))
                    put(CremaContract.STATE_FILE, sha256(state))
                    put("crema-cqt-h1.bin", sha256(planH1))
                    put("crema-cqt-h2.bin", sha256(planH2))
                })
                RecognizerRunner(
                    CremaStreamingRecognizer(
                        modelBytes = model,
                        runtimeStateJson = state.toString(Charsets.UTF_8),
                        harmonic1PlanBytes = planH1,
                        harmonic2PlanBytes = planH2,
                        dspRescue = engine.dspRescue,
                        rescueConfidenceThreshold = engine.rescueConfidenceThreshold,
                    )
                )
            }

            Engine.BTC, Engine.BTC_NO_RESCUE, Engine.BTC_RESCUE -> {
                val model = File(assets, "btc/${BtcContract.MODEL_FILE}").readBytes()
                val metaBytes = File(assets, "btc/${BtcContract.METADATA_FILE}").readBytes()
                val plan = File(assets, "btc/${BtcContract.PLAN_FILE}").readBytes()
                val metadata = BtcMetadata.parse(metaBytes)
                // BtcOnnxRunner already fails the construction unless the model hashes
                // to the value the metadata carries, and the recognizer's own init
                // checks the CQT plan against BtcContract.PLAN_SHA256; both are asserted
                // here too so a swapped asset is caught before any audio is fed.
                require(sha256(model) == metadata.modelSha256) { "Unexpected BTC model SHA-256" }
                require(sha256(plan) == BtcContract.PLAN_SHA256) { "Unexpected BTC CQT plan SHA-256" }
                provenance.put("model_sha256", JSONObject().apply {
                    put(BtcContract.MODEL_FILE, sha256(model))
                    put(BtcContract.METADATA_FILE, sha256(metaBytes))
                    put(BtcContract.PLAN_FILE, sha256(plan))
                })
                provenance.put("btc_source_commit", metadata.sourceCommit)
                provenance.put("btc_checkpoint_blob_sha", metadata.checkpointBlobSha)
                RecognizerRunner(
                    BtcStreamingRecognizer(
                        modelBytes = model,
                        metadata = metadata,
                        cqtPlanBytes = plan,
                        dspRescue = engine.dspRescue,
                        rescueConfidenceThreshold = engine.rescueConfidenceThreshold,
                    )
                )
            }

            Engine.SOLITITO, Engine.SOLITITO_GATE34 -> {
                val model = File(assets, "solitito/${SolititoContract.MODEL_FILE}").readBytes()
                val plan = File(assets, "solitito/${SolititoContract.PLAN_FILE}").readBytes()
                require(sha256(model) == SolititoContract.MODEL_SHA256) {
                    "Unexpected solitito model SHA-256"
                }
                require(sha256(plan) == SolititoContract.PLAN_SHA256) {
                    "Unexpected solitito DSP plan SHA-256"
                }
                provenance.put("model_sha256", JSONObject().apply {
                    put(SolititoContract.MODEL_FILE, sha256(model))
                    put(SolititoContract.PLAN_FILE, sha256(plan))
                })
                provenance.put("solitito_source_commit", SolititoContract.SOURCE_COMMIT)
                provenance.put(
                    "solitito_upstream_weights_sha256",
                    SolititoContract.UPSTREAM_WEIGHTS_SHA256,
                )
                RecognizerRunner(
                    SolititoStreamingRecognizer(
                        modelBytes = model,
                        planBytes = plan,
                        gateDb = engine.gateDb ?: SolititoContract.GATE_DB,
                    )
                )
            }

            Engine.CLASSIC -> {
                provenance.put("model_sha256", JSONObject())
                ClassicRunner(
                    PitchAnalyzer(
                        profile = InstrumentProfile.Guitar,
                        mode = DetectionMode.CHORD,
                    )
                )
            }
        }

    fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    /** Summary the test prints so the Python driver can confirm what ran. */
    fun summary(results: List<Result>): String = JSONArray().apply {
        for (result in results) {
            put(JSONObject().apply {
                put("engine", result.engine.id)
                put("item", result.item.id)
                put("trace", result.trace.absolutePath)
                put("meta", result.meta.absolutePath)
                put("rows", result.rows)
            })
        }
    }.toString()
}
