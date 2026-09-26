package com.nicos.pitchkit.tuner.harmony.animechord

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Opt-in whole-song benchmark for the anime engine on the JVM.
 *
 * This engine's cost is the reason it is an Advanced pick rather than part of Auto, so the
 * cost has to be measurable without a phone. Nothing runs unless the caller points it at
 * audio:
 *
 * ```
 * gradle :PitchKit:testDebugUnitTest --tests '*AnimeChordSongBenchmarkTest' \
 *     -Danimechord.bench.audio=C:\path\song.s16 -Danimechord.bench.rate=44100
 * ```
 *
 * The file is raw little-endian int16 mono PCM at `animechord.bench.rate`, which is what
 * `SongFileAnalyzer` hands the analyzer after MediaCodec. Several files can be given,
 * separated by `;`. Heap is reported from the JVM itself; the native ONNX Runtime arena —
 * the number that actually decides whether this fits on a device — is outside the heap and
 * has to be read from the process's working set while this runs.
 */
class AnimeChordSongBenchmarkTest {
    @Test
    fun benchmark() {
        val paths = System.getProperty("animechord.bench.audio").orEmpty()
        assumeTrue("set -Danimechord.bench.audio to run the benchmark", paths.isNotBlank())
        val rate = System.getProperty("animechord.bench.rate", "44100").toInt()
        val chunk = System.getProperty("animechord.bench.chunk", "8192").toInt()

        val model = AnimeChordTestAssets.read(AnimeChordContract.MODEL_FILE)
        val plan = AnimeChordTestAssets.read(AnimeChordContract.CQT_PLAN_FILE)
        val vocabulary = AnimeChordTestAssets.read(AnimeChordContract.VOCABULARY_FILE)
            .toString(Charsets.UTF_8)

        for (path in paths.split(';').map { it.trim() }.filter { it.isNotEmpty() }) {
            val file = File(path)
            val samples = readPcm(file)
            val seconds = samples.size.toDouble() / rate
            val runtime = Runtime.getRuntime()
            runtime.gc()
            val heapBefore = runtime.totalMemory() - runtime.freeMemory()

            val started = System.nanoTime()
            val analysis = AnimeChordSongAnalyzer(model, plan, vocabulary).use { analyzer ->
                var offset = 0
                while (offset < samples.size) {
                    val end = minOf(samples.size, offset + chunk)
                    analyzer.accept(samples.copyOfRange(offset, end), rate)
                    offset = end
                }
                analyzer.finish((seconds * 1000).toLong())
            }
            val wall = (System.nanoTime() - started) / 1e9
            val heapPeak = runtime.totalMemory() - runtime.freeMemory()

            println(
                "animechord bench ${file.name}: audio=${"%.1f".format(seconds)}s " +
                    "wall=${"%.1f".format(wall)}s (${"%.2f".format(seconds / wall)}x realtime) " +
                    "chords=${analysis.chords.size} sections=${analysis.sections.size} " +
                    "heap=${(heapPeak - heapBefore) / 1_000_000} MB " +
                    "committedHeap=${runtime.totalMemory() / 1_000_000} MB"
            )
            println(
                "animechord bench ${file.name} first chords: " +
                    analysis.chords.take(8).joinToString(", ") {
                        "${it.label}@${it.startMs}ms"
                    }
            )
        }
    }

    private fun readPcm(file: File): FloatArray {
        val buffer = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        return FloatArray(buffer.remaining()) { buffer.get(it) / 32768.0f }
    }
}
