package com.nicos.pitchkit.tuner.harmony.crema

import android.content.Context
import java.security.MessageDigest

object CremaAndroidFactory {
    private const val ASSET_DIRECTORY = "crema"
    private const val PLAN_H1_FILE = "crema-cqt-h1.bin"
    private const val PLAN_H2_FILE = "crema-cqt-h2.bin"

    fun assetsInstalled(context: Context): Boolean {
        val names = context.assets.list(ASSET_DIRECTORY)?.toSet().orEmpty()
        return CremaContract.MODEL_FILE in names &&
            CremaContract.STATE_FILE in names &&
            PLAN_H1_FILE in names &&
            PLAN_H2_FILE in names
    }

    fun create(
        context: Context,
        referenceA4Hz: Double = 440.0,
        preferFlats: Boolean = false,
    ): CremaStreamingRecognizer {
        require(assetsInstalled(context)) {
            "Crema assets are not installed. Run tools/fetch-crema.ps1 first."
        }

        val model = readBytes(context, CremaContract.MODEL_FILE)
        val stateBytes = readBytes(context, CremaContract.STATE_FILE)
        val planH1 = readBytes(context, PLAN_H1_FILE)
        val planH2 = readBytes(context, PLAN_H2_FILE)

        verify("model", model, CremaContract.MODEL_SIZE, CremaContract.MODEL_SHA256)
        verify("runtime state", stateBytes, CremaContract.STATE_SIZE, CremaContract.STATE_SHA256)
        verify("harmonic-1 CQT plan", planH1, 45_568, CremaContract.PLAN_H1_SHA256)
        verify("harmonic-2 CQT plan", planH2, 45_568, CremaContract.PLAN_H2_SHA256)

        return CremaStreamingRecognizer(
            modelBytes = model,
            runtimeStateJson = stateBytes.toString(Charsets.UTF_8),
            harmonic1PlanBytes = planH1,
            harmonic2PlanBytes = planH2,
            referenceA4Hz = referenceA4Hz,
            preferFlats = preferFlats,
        )
    }

    private fun readBytes(context: Context, file: String): ByteArray = context.assets
        .open("$ASSET_DIRECTORY/$file")
        .use { it.readBytes() }

    private fun verify(label: String, bytes: ByteArray, expectedSize: Int, expectedSha: String) {
        val actualSha = sha256(bytes)
        require(bytes.size == expectedSize && actualSha == expectedSha) {
            "Unexpected Crema $label: size=${bytes.size} sha256=$actualSha"
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
