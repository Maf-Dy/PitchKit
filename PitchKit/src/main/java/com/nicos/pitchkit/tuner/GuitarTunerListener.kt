package com.nicos.pitchkit.tuner

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.nicos.pitchkit.BuildConfig
import com.nicos.pitchkit.tuner.models.InstrumentProfile

/**
 * Android/Compose microphone convenience wrapper around [PitchAnalyzer].
 *
 * Use [PitchAnalyzer] directly when the PCM source is a file, USB device,
 * playback capture, Media3 pipeline, or anything other than this app's mic.
 */
@Composable
fun GuitarTunerListener(
    profile: InstrumentProfile = InstrumentProfile.Guitar,
    mode: DetectionMode = DetectionMode.AUTO,
    referenceA4Hz: Double = 440.0,
    highPassCutoffHz: Double = 30.0,
    autoChordThreshold: Double = 0.30,
    chordMinScore: Double = 0.20,
    titleText: String = "Microphone needed",
    permanentlyDeniedText: String = "Microphone access is blocked. Please enable it in Settings.",
    rationaleText: String = "This app needs microphone access to detect notes and chords.",
    openSettingsText: String = "Open Settings",
    allowText: String = "Allow",
    dismissText: String = "Not now",
    onResult: (TuningResult) -> Unit,
) {
    val context = LocalContext.current
    val activity = context as? Activity

    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO,
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    var showDialog by remember { mutableStateOf(false) }
    var permanentlyDenied by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        granted = isGranted
        if (!isGranted) {
            permanentlyDenied = activity?.let {
                !ActivityCompat.shouldShowRequestPermissionRationale(
                    it,
                    Manifest.permission.RECORD_AUDIO,
                )
            } ?: false
            showDialog = true
        }
    }

    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                granted = ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.RECORD_AUDIO,
                ) == PackageManager.PERMISSION_GRANTED
                if (granted) showDialog = false
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    LaunchedEffect(Unit) {
        if (!granted) launcher.launch(Manifest.permission.RECORD_AUDIO)
    }

    if (granted) {
        val engine = remember(
            profile,
            mode,
            referenceA4Hz,
            highPassCutoffHz,
            autoChordThreshold,
            chordMinScore,
        ) {
            TunerEngine(
                profile = profile,
                mode = mode,
                referenceA4Hz = referenceA4Hz,
                highPassCutoffHz = highPassCutoffHz,
                autoChordThreshold = autoChordThreshold,
                chordMinScore = chordMinScore,
            )
        }

        LaunchedEffect(engine) {
            engine.start().collect { result ->
                if (BuildConfig.DEBUG) {
                    val debugValue = when (result) {
                        is TuningResult.Note ->
                            "${result.name} ${result.freq} (${"%.0f".format(result.cents)}c)"
                        is TuningResult.Chord -> result.name
                        TuningResult.Silence -> "-"
                    }
                    Log.d("PitchKit", debugValue)
                }
                onResult(result)
            }
        }
    }

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(titleText) },
            text = {
                Text(if (permanentlyDenied) permanentlyDeniedText else rationaleText)
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showDialog = false
                        if (permanentlyDenied) {
                            val intent = Intent(
                                Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                                Uri.fromParts("package", context.packageName, null),
                            )
                            context.startActivity(intent)
                        } else {
                            launcher.launch(Manifest.permission.RECORD_AUDIO)
                        }
                    }
                ) {
                    Text(if (permanentlyDenied) openSettingsText else allowText)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDialog = false }) {
                    Text(dismissText)
                }
            },
        )
    }
}
