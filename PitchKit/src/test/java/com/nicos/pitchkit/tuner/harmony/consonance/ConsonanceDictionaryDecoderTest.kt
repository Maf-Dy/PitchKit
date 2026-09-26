package com.nicos.pitchkit.tuner.harmony.consonance

import kotlin.math.ln
import org.junit.Assert.assertEquals
import org.junit.Test

class ConsonanceDictionaryDecoderTest {
    private val cMajor = candidate("C:maj", root = 0, bass = 0, pitches = intArrayOf(0, 4, 7))
    private val fMajor = candidate("F:maj", root = 5, bass = 5, pitches = intArrayOf(5, 9, 0))
    private val gMajor = candidate("G:maj", root = 7, bass = 7, pitches = intArrayOf(7, 11, 2))
    private val noChord = ConsonanceDictionaryCandidate(
        rawLabel = "N",
        displayLabel = null,
        root = 12,
        bass = 12,
        mask = 0,
    )
    private val candidates = listOf(noChord, cMajor, fMajor, gMajor)

    @Test
    fun groupsClearChordsIntoSegments() {
        val heads = heads(List(10) { cMajor } + List(6) { fMajor } + List(4) { gMajor })
        val segments = decoder().decode(heads)

        assertEquals(listOf("C:maj", "F:maj", "G:maj"), segments.map { it.candidate.rawLabel })
        assertEquals(listOf(0, 10, 16), segments.map { it.startFrame })
        assertEquals(listOf(10, 16, 20), segments.map { it.endFrame })
    }

    @Test
    fun absorbsSingleFrameFlipUnderThePenalty() {
        val frames = MutableList(20) { cMajor }
        frames[9] = gMajor
        val heads = heads(frames)

        assertEquals(
            listOf("C:maj"),
            decoder(penalty = 100.0).decode(heads).map { it.candidate.rawLabel },
        )
        assertEquals(
            listOf("C:maj", "G:maj", "C:maj"),
            decoder(penalty = 0.0).decode(heads).map { it.candidate.rawLabel },
        )
    }

    @Test
    fun keepsTheCurrentStateWhenScoresTie() {
        // Frames 0..8 are ambiguous: the three triads are exactly tied every frame, so
        // only the last frame - which clearly selects G - decides the winner. Because
        // switching requires a strictly better score, G's own history is kept and the
        // whole song is one G segment; switching on equality would splice in the
        // lower-indexed tied candidate instead.
        val ambiguous = ambiguousHeads(9)
        val decisive = heads(List(1) { gMajor })
        val heads = ConsonanceHeads(
            frames = 10,
            root = ambiguous.root + decisive.root,
            bass = ambiguous.bass + decisive.bass,
            pitch = ambiguous.pitch + decisive.pitch,
        )

        val segments = decoder(penalty = 0.0).decode(heads)
        assertEquals(listOf("G:maj"), segments.map { it.candidate.rawLabel })
        assertEquals(0, segments.single().startFrame)
        assertEquals(10, segments.single().endFrame)
    }

    /**
     * The competitive third softmax asks which third the model prefers instead of rewarding
     * a sus4 for the absence of one. With a merely weak third (0.2 major, 0.1 minor) the
     * major triad now outscores the sus4 even though the 4th is the more active pitch class.
     */
    @Test
    fun prefersTheMajorTriadOverSusWhenTheThirdIsOnlyWeak() {
        val dictionary = susDictionary()
        val heads = pitchHeads(thirdMajor = 0.2, thirdMinor = 0.1, fourth = 0.55)

        assertEquals(
            listOf("C:maj"),
            ConsonanceDictionaryDecoder(dictionary, dictionary.transitionPenalty)
                .decode(heads)
                .map { it.candidate.rawLabel },
        )
    }

    @Test
    fun stillPrefersSusWhenNoThirdIsPresentAtAll() {
        val dictionary = susDictionary()
        val heads = pitchHeads(thirdMajor = 0.02, thirdMinor = 0.01, fourth = 0.55)

        assertEquals(
            listOf("C:sus4"),
            ConsonanceDictionaryDecoder(dictionary, dictionary.transitionPenalty)
                .decode(heads)
                .map { it.candidate.rawLabel },
        )
    }

    /**
     * The Bernoulli lane has no competitive third softmax, so it still rewards the sus4 for
     * the absent third and the active 4th where the competitive lane prefers the triad.
     * Cross-checked against consonance_dictionary_decode.py --likelihood bernoulli.
     */
    @Test
    fun theBernoulliLanePrefersSusWhereCompetitivePrefersTheTriad() {
        val dictionary = susDictionary()
        val heads = pitchHeads(thirdMajor = 0.2, thirdMinor = 0.1, fourth = 0.55)

        assertEquals(
            listOf("C:sus4"),
            ConsonanceDictionaryDecoder(
                dictionary = dictionary,
                transitionPenalty = dictionary.transitionPenalty,
                likelihood = ConsonanceLikelihood.BERNOULLI,
            ).decode(heads).map { it.candidate.rawLabel },
        )
    }

    /**
     * The experimental bass rule: with a flat pitch head the two triads score identically
     * on pitch, the root head mildly prefers C and the bass head is sure the bass is G.
     * Competitive follows the bass head onto G:maj; competitive-bass lets the root-position
     * C:maj keep half the credit of its own fifth, which is enough to hold the root.
     * Cross-checked against consonance_dictionary_decode.py --likelihood competitive-bass.
     */
    @Test
    fun theCompetitiveBassLaneHoldsTheRootAgainstAPassingBassNote() {
        val dictionary = ConsonanceDictionary(
            transitionPenalty = 100.0,
            groupNonePriors = GROUP_NONE_PRIORS,
            candidates = listOf(noChord, cMajor, gMajor),
        )
        val frames = 4
        val root = FloatArray(frames * ConsonanceContract.ROOT_COUNT)
        val bass = FloatArray(frames * ConsonanceContract.BASS_COUNT)
        for (frame in 0 until frames) {
            root[frame * ConsonanceContract.ROOT_COUNT] = 1f
            bass[frame * ConsonanceContract.BASS_COUNT + 7] = 8f
        }
        val heads = ConsonanceHeads(
            frames = frames,
            root = root,
            bass = bass,
            pitch = FloatArray(frames * ConsonanceContract.PITCH_COUNT),
        )

        fun decodeWith(likelihood: ConsonanceLikelihood) = ConsonanceDictionaryDecoder(
            dictionary = dictionary,
            transitionPenalty = dictionary.transitionPenalty,
            likelihood = likelihood,
        ).decode(heads).map { it.candidate.rawLabel }

        assertEquals(listOf("G:maj"), decodeWith(ConsonanceLikelihood.COMPETITIVE))
        assertEquals(listOf("C:maj"), decodeWith(ConsonanceLikelihood.COMPETITIVE_BASS))
    }

    private fun susDictionary() = ConsonanceDictionary(
        transitionPenalty = 100.0,
        groupNonePriors = GROUP_NONE_PRIORS,
        candidates = listOf(
            noChord,
            cMajor,
            candidate("C:sus4", root = 0, bass = 0, pitches = intArrayOf(0, 5, 7)),
        ),
    )

    /** Four identical frames stating C and G clearly plus the given third / fourth activations. */
    private fun pitchHeads(thirdMajor: Double, thirdMinor: Double, fourth: Double): ConsonanceHeads {
        val frames = 4
        val root = FloatArray(frames * ConsonanceContract.ROOT_COUNT)
        val bass = FloatArray(frames * ConsonanceContract.BASS_COUNT)
        val probabilities = DoubleArray(12) { 0.02 }
        probabilities[0] = 0.95
        probabilities[7] = 0.95
        probabilities[3] = thirdMinor
        probabilities[4] = thirdMajor
        probabilities[5] = fourth
        val pitch = FloatArray(frames * ConsonanceContract.PITCH_COUNT) {
            logit(probabilities[it % ConsonanceContract.PITCH_COUNT])
        }
        for (frame in 0 until frames) {
            root[frame * ConsonanceContract.ROOT_COUNT] = 8f
            bass[frame * ConsonanceContract.BASS_COUNT] = 8f
        }
        return ConsonanceHeads(frames, root, bass, pitch)
    }

    private fun logit(probability: Double): Float = ln(probability / (1.0 - probability)).toFloat()

    private fun decoder(penalty: Double = 100.0) = ConsonanceDictionaryDecoder(
        ConsonanceDictionary(
            transitionPenalty = penalty,
            groupNonePriors = GROUP_NONE_PRIORS,
            candidates = candidates,
        ),
        penalty,
    )

    private fun candidate(
        label: String,
        root: Int,
        bass: Int,
        pitches: IntArray,
    ) = ConsonanceDictionaryCandidate(
        rawLabel = label,
        displayLabel = label,
        root = root,
        bass = bass,
        mask = pitches.fold(0) { mask, pc -> mask or (1 shl pc) },
    )

    private fun heads(frames: List<ConsonanceDictionaryCandidate>): ConsonanceHeads {
        val root = FloatArray(frames.size * ConsonanceContract.ROOT_COUNT)
        val bass = FloatArray(frames.size * ConsonanceContract.BASS_COUNT)
        val pitch = FloatArray(frames.size * ConsonanceContract.PITCH_COUNT) { -4f }
        frames.forEachIndexed { frame, candidate ->
            root[frame * ConsonanceContract.ROOT_COUNT + candidate.root] = 8f
            bass[frame * ConsonanceContract.BASS_COUNT + candidate.bass] = 8f
            for (pc in 0 until 12) {
                if (candidate.mask and (1 shl pc) != 0) {
                    pitch[frame * ConsonanceContract.PITCH_COUNT + pc] = 4f
                }
            }
        }
        return ConsonanceHeads(frames.size, root, bass, pitch)
    }

    /** Uniform heads: every three-note triad scores identically, N scores lower. */
    private fun ambiguousHeads(frames: Int): ConsonanceHeads = ConsonanceHeads(
        frames = frames,
        root = FloatArray(frames * ConsonanceContract.ROOT_COUNT),
        bass = FloatArray(frames * ConsonanceContract.BASS_COUNT),
        pitch = FloatArray(frames * ConsonanceContract.PITCH_COUNT) { 1f },
    )

    private companion object {
        /** The values consonance-dictionary.json carries, in ConsonanceGroups.NAMES order. */
        val GROUP_NONE_PRIORS = doubleArrayOf(0.13157894736842105, 0.0, 0.4298245614035088)
    }
}
