package com.nicos.pitchkit.tuner.harmony.crema

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CremaInferencePlannerTest {
    @Test
    fun songLengthSequenceRunsInASinglePass() {
        // 2,000 frames is ~3.1 min at hop 4096 / 44.1 kHz, i.e. a normal song.
        val plan = CremaInferencePlanner.plan(SONG_FRAMES)

        assertEquals(1, plan.size)
        assertEquals(CremaInferenceChunk(0, SONG_FRAMES, 0, SONG_FRAMES), plan.single())
    }

    @Test
    fun singlePassFramesDecodeIntoAValidSegmentList() {
        val state = CremaRuntimeState(
            labels = listOf("N", "C:maj", "G:maj"),
            transitionDiagonal = 0.95,
            transitionOffDiagonal = 0.01,
        )
        val decoder = CremaSongViterbiDecoder(state, preferFlats = false)
        val chunk = CremaInferencePlanner.plan(SONG_FRAMES).single()
        val heads = syntheticHeads(chunk.frameCount, switchAt = SONG_FRAMES / 2)

        // Mirrors CremaSongAnalyzer.inferChunks.
        for (frame in chunk.commitStart until chunk.commitEnd) {
            decoder.add(
                heads = heads,
                localFrame = frame - chunk.startFrame,
                globalFrame = frame.toLong(),
            )
        }
        val predictions = decoder.decode()

        assertEquals(SONG_FRAMES, predictions.size)
        predictions.forEachIndexed { index, prediction ->
            assertEquals(index.toLong(), prediction.frame)
            assertTrue(prediction.confidence in 0.0..1.0)
        }
        val labels = predictions.map { it.label }
        assertEquals("C", labels.first())
        assertEquals("G", labels.last())
        assertEquals(listOf("C", "G"), labels.distinct())
    }

    @Test
    fun overlongSequenceFallsBackToContiguousChunks() {
        val frames = CremaInferencePlanner.MAX_SINGLE_PASS_FRAMES + 1
        val plan = CremaInferencePlanner.plan(frames)

        assertTrue(plan.size > 1)
        assertEquals(0, plan.first().commitStart)
        assertEquals(frames, plan.last().commitEnd)
        plan.zipWithNext { previous, next ->
            assertEquals(previous.commitEnd, next.commitStart)
        }
        plan.forEach { chunk ->
            assertTrue(chunk.frameCount <= CremaInferencePlanner.FALLBACK_CHUNK_FRAMES)
            assertTrue(chunk.commitStart >= chunk.startFrame)
            assertTrue(chunk.commitEnd <= chunk.endFrame)
        }
    }

    private fun syntheticHeads(frames: Int, switchAt: Int): CremaHeads {
        val tag = FloatArray(frames * CremaContract.CHORD_COUNT) { 0.01f }
        val bass = FloatArray(frames * CremaContract.BASS_COUNT) { 0.05f }
        for (frame in 0 until frames) {
            val chordIndex = if (frame < switchAt) 1 else 2
            tag[frame * CremaContract.CHORD_COUNT + chordIndex] = 0.9f
            // Root position: C for C:maj, G for G:maj.
            bass[frame * CremaContract.BASS_COUNT + if (frame < switchAt) 0 else 7] = 0.9f
        }
        return CremaHeads(
            frames = frames,
            tag = tag,
            pitch = FloatArray(frames * CremaContract.PITCH_COUNT),
            root = FloatArray(frames * CremaContract.ROOT_COUNT),
            bass = bass,
        )
    }

    private companion object {
        const val SONG_FRAMES = 2_000
    }
}
