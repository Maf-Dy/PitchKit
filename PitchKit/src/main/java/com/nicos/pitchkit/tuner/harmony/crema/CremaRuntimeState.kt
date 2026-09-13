package com.nicos.pitchkit.tuner.harmony.crema

import org.json.JSONObject

internal data class CremaRuntimeState(
    val labels: List<String>,
    val transitionDiagonal: Double,
    val transitionOffDiagonal: Double,
) {
    companion object {
        fun parse(json: String): CremaRuntimeState {
            val root = JSONObject(json)
            require(root.getInt("schema_version") == 1)

            val source = root.getJSONObject("source")
            require(source.getString("name") == "crema")
            require(source.getString("version") == "0.2.0")

            val preprocessing = root.getJSONObject("preprocessing")
            require(preprocessing.getInt("sample_rate") == CremaContract.SAMPLE_RATE)
            require(preprocessing.getInt("hop_length") == CremaContract.HOP_LENGTH)
            require(preprocessing.getInt("octaves") == 6)
            require(preprocessing.getInt("oversample") == 3)
            val harmonics = preprocessing.getJSONArray("harmonics")
            require(harmonics.length() == 2 && harmonics.getInt(0) == 1 && harmonics.getInt(1) == 2)

            val decoder = root.getJSONObject("decoder")
            require(decoder.getInt("sample_rate") == CremaContract.SAMPLE_RATE)
            require(decoder.getInt("hop_length") == CremaContract.HOP_LENGTH)
            val labelsJson = decoder.getJSONArray("labels")
            require(labelsJson.length() == CremaContract.CHORD_COUNT)
            val labels = List(labelsJson.length()) { labelsJson.getString(it) }

            val transition = decoder.getJSONObject("transition")
            require(transition.getString("encoding") == "uniform-off-diagonal")
            val shape = transition.getJSONArray("shape")
            require(shape.getInt(0) == CremaContract.CHORD_COUNT)
            require(shape.getInt(1) == CremaContract.CHORD_COUNT)

            return CremaRuntimeState(
                labels = labels,
                transitionDiagonal = transition.getDouble("diagonal"),
                transitionOffDiagonal = transition.getDouble("off_diagonal"),
            )
        }
    }
}
