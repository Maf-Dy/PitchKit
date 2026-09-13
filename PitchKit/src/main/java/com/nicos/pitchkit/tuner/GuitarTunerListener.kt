package com.nicos.pitchkit.tuner

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaRecorder
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
import androidx.lifecycle.repeatOnLifecycle
import com.nicos.pitchkit.BuildConfig
import com.nicos.pitchkit.tuner.harmony.ChordRecognizer
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetAndroidFactory
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetContract
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetStreamingRecognizer
import com.nicos.pitchkit.tuner.harmony.crema.CremaAndroidFactory
import com.nicos.pitchkit.tuner.harmony.crema.CremaContract
import com.nicos.pitchkit.tuner.harmony.crema.CremaStreamingRecognizer
import com.nicos.pitchkit.tuner.models.InstrumentProfile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@Composable
fun GuitarTunerListener(
    profile: InstrumentProfile = InstrumentProfile.Guitar,
    mode: DetectionMode = DetectionMode.AUTO,
    referenceA4Hz: Double = 440.0,
    preferFlats: Boolean = false,
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
    val applicationContext = context.applicationContext
    val activity = context as? Activity

    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    var showDialog by remember { mutableStateOf(false) }
    var permanentlyDenied by remember { mutableStateOf(false) }

    val cremaAssetsInstalled = remember(applicationContext) {
        CremaAndroidFactory.assetsInstalled(applicationContext)
    }
    val chordNetAssetsInstalled = remember(applicationContext) {
        ChordNetAndroidFactory.assetsInstalled(applicationContext)
    }
    val neuralAssetsInstalled = cremaAssetsInstalled || chordNetAssetsInstalled

    var neuralRecognizer by remember { mutableStateOf<ChordRecognizer?>(null) }
    var neuralLoadFailed by remember { mutableStateOf(false) }

    LaunchedEffect(
        mode,
        cremaAssetsInstalled,
        chordNetAssetsInstalled,
        referenceA4Hz,
        preferFlats,
    ) {
        if (mode != DetectionMode.CHORD || !neuralAssetsInstalled) {
            neuralRecognizer = null
            neuralLoadFailed = false
            return@LaunchedEffect
        }

        neuralLoadFailed = false
        neuralRecognizer = null
        neuralRecognizer = withContext(Dispatchers.IO) {
            if (cremaAssetsInstalled) {
                try {
                    CremaAndroidFactory.create(
                        context = applicationContext,
                        referenceA4Hz = referenceA4Hz,
                        preferFlats = preferFlats,
                    ).also {
                        if (BuildConfig.DEBUG) Log.d("PitchKit", "Crema neural recognizer loaded")
                    }
                } catch (error: Throwable) {
                    Log.e("PitchKit", "Crema failed to load; using classic DSP fallback (not ChordNet)", error)
                    null
                }
            } else if (chordNetAssetsInstalled) {
                try {
                    ChordNetAndroidFactory.create(
                        context = applicationContext,
                        referenceA4Hz = referenceA4Hz,
                    ).also {
                        if (BuildConfig.DEBUG) Log.d("PitchKit", "ChordNet fallback recognizer loaded")
                    }
                } catch (error: Throwable) {
                    Log.e("PitchKit", "ChordNet failed to load; using classic fallback", error)
                    null
                }
            } else {
                null
            }
        }
        neuralLoadFailed = neuralRecognizer == null
    }

    DisposableEffect(neuralRecognizer) {
        val recognizer = neuralRecognizer
        onDispose { recognizer?.close() }
    }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        granted = isGranted
        if (!isGranted) {
            permanentlyDenied = activity?.let {
                !ActivityCompat.shouldShowRequestPermissionRationale(it, Manifest.permission.RECORD_AUDIO)
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

    val waitingForNeural = mode == DetectionMode.CHORD &&
        neuralAssetsInstalled &&
        neuralRecognizer == null &&
        !neuralLoadFailed

    if (granted && !waitingForNeural) {
        val engine = remember(
            profile,
            mode,
            referenceA4Hz,
            highPassCutoffHz,
            autoChordThreshold,
            chordMinScore,
            neuralRecognizer,
        ) {
            val neural = neuralRecognizer
            val neuralSampleRate = when (neural) {
                is CremaStreamingRecognizer -> CremaContract.SAMPLE_RATE
                is ChordNetStreamingRecognizer -> ChordNetContract.SAMPLE_RATE
                else -> 44100
            }
            val engineBufferSize = when (neural) {
                is CremaStreamingRecognizer -> CremaContract.HOP_LENGTH
                is ChordNetStreamingRecognizer -> ChordNetContract.HOP_LENGTH
                else -> if (mode == DetectionMode.NOTE) 4096 else 8192
            }

            TunerEngine(
                profile = profile,
                mode = mode,
                referenceA4Hz = referenceA4Hz,
                highPassCutoffHz = highPassCutoffHz,
                autoChordThreshold = autoChordThreshold,
                chordMinScore = chordMinScore,
                chordRecognizer = neural,
                preferredSampleRate = neuralSampleRate,
                bufferSize = engineBufferSize,
                preferredAudioSource = if (neural != null) {
                    MediaRecorder.AudioSource.VOICE_RECOGNITION
                } else {
                    MediaRecorder.AudioSource.MIC
                },
            )
        }

        DisposableEffect(engine) {
            onDispose { engine.close() }
        }

        LaunchedEffect(engine, lifecycleOwner) {
            lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                engine.start().collect { result ->
                    if (BuildConfig.DEBUG) Log.d("PitchKit", result.toString())
                    onResult(result)
                }
            }
        }
    }

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(titleText) },
            text = { Text(if (permanentlyDenied) permanentlyDeniedText else rationaleText) },
            confirmButton = {
                TextButton(onClick = {
                    showDialog = false
                    if (permanentlyDenied) {
                        context.startActivity(
                            Intent(
                                Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                                Uri.fromParts("package", context.packageName, null),
                            )
                        )
                    } else {
                        launcher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                }) {
                    Text(if (permanentlyDenied) openSettingsText else allowText)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDialog = false }) { Text(dismissText) }
            },
        )
    }
}
