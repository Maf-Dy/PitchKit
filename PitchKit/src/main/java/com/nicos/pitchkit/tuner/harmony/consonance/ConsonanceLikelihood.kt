package com.nicos.pitchkit.tuner.harmony.consonance

/**
 * Observation-likelihood variants of the Consonance dictionary Viterbi, mirroring the
 * `--likelihood` lanes of tools/accuracy_audit/consonance_dictionary_decode.py.
 *
 * [id] is the exact spelling the Python reference, the exported dictionary asset and the
 * review tool use, so a recorded backend string names the same lane on both sides.
 */
enum class ConsonanceLikelihood(val id: String, val displayName: String) {
    /** Independent Bernoulli term over all twelve pitch classes. The original lane. */
    BERNOULLI("bernoulli", "Bernoulli"),

    /** Mutually exclusive third / fifth / seventh softmaxes. The recommended default. */
    COMPETITIVE("competitive", "Competitive"),

    /** Competitive plus the experimental root-position bass allowance. */
    COMPETITIVE_BASS("competitive-bass", "Competitive + bass"),
    ;

    companion object {
        fun fromId(id: String): ConsonanceLikelihood? = entries.firstOrNull { it.id == id }

        fun requireFromId(id: String): ConsonanceLikelihood = requireNotNull(fromId(id)) {
            "Unsupported Consonance likelihood '$id'; expected one of " +
                entries.joinToString(", ") { it.id }
        }
    }
}

/**
 * Chord-change penalties the review tool sweeps, in log-likelihood units. The decoder
 * accepts any positive value; this list is only what the UI offers.
 */
object ConsonanceTransitionPenalties {
    val OFFERED = intArrayOf(10, 30, 60, 100, 120, 150, 250)
}
