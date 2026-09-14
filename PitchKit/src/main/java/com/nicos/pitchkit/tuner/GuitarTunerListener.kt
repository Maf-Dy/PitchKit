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
import androidx.compose.runtime.rememberUpdatedState
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
    chordEngine: ChordEngine = ChordEngine.AUTO,
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
    val currentOnResult by rememberUpdatedState(onResult)

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
    val selectedNeuralAssetsInstalled = when (chordEngine) {
        ChordEngine.AUTO -> cremaAssetsInstalled || chordNetAssetsInstalled
        ChordEngine.CREMA -> cremaAssetsInstalled
        ChordEngine.CHORD_NET -> chordNetAssetsInstalled
        ChordEngine.CLASSIC -> false
    }

    var neuralRecognizer by remember { mutableStateOf<ChordRecognizer?>(null) }
    var neuralRecognizerSelection by remember { mutableStateOf<ChordEngine?>(null) }
    var neuralLoadFailedSelection by remember { mutableStateOf<ChordEngine?>(null) }

    LaunchedEffect(
        mode,
        chordEngine,
        cremaAssetsInstalled,
        chordNetAssetsInstalled,
        referenceA4Hz,
        preferFlats,
    ) {
        if (mode != DetectionMode.CHORD || chordEngine == ChordEngine.CLASSIC) {
            neuralRecognizer = null
            neuralRecognizerSelection = null
            neuralLoadFailedSelection = null
            return@LaunchedEffect
        }

        val requestedSelection = chordEngine
        neuralRecognizer = null
        neuralRecognizerSelection = null
        neuralLoadFailedSelection = null

        val loaded = withContext(Dispatchers.IO) {
            fun tryCrema(): ChordRecognizer? {
                if (!cremaAssetsInstalled) {
                    if (BuildConfig.DEBUG) Log.d("PitchKit", "Crema assets are not installed")
                    return null
                }
                return try {
                    CremaAndroidFactory.create(
                        context = applicationContext,
                        referenceA4Hz = referenceA4Hz,
                        preferFlats = preferFlats,
                    ).also {
                        if (BuildConfig.DEBUG) Log.d("PitchKit", "Crema neural recognizer loaded")
                    }
                } catch (error: Throwable) {
                    Log.e("PitchKit", "Crema failed to load", error)
                    null
                }
            }

            fun tryChordNet(): ChordRecognizer? {
                if (!chordNetAssetsInstalled) {
                    if (BuildConfig.DEBUG) Log.d("PitchKit", "ChordNet assets are not installed")
                    return null
                }
                return try {
                    ChordNetAndroidFactory.create(
                        context = applicationContext,
                        referenceA4Hz = referenceA4Hz,
                    ).also {
                        if (BuildConfig.DEBUG) Log.d("PitchKit", "ChordNet neural recognizer loaded")
                    }
                } catch (error: Throwable) {
                    Log.e("PitchKit", "ChordNet failed to load", error)
                    null
                }
            }

            when (requestedSelection) {
                ChordEngine.AUTO -> tryChordNet() ?: tryCrema()
                ChordEngine.CREMA -> tryCrema()
                ChordEngine.CHORD_NET -> tryChordNet()
                ChordEngine.CLASSIC -> null
            }
        }

        neuralRecognizer = loaded
        neuralRecognizerSelection = if (loaded != null) requestedSelection else null
        neuralLoadFailedSelection = if (loaded == null) requestedSelection else null
        if (loaded == null && BuildConfig.DEBUG) {
            Log.d("PitchKit", "${requestedSelection.name} unavailable; using Classic DSP")
        }
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

    val activeNeuralRecognizer = neuralRecognizer.takeIf {
        mode == DetectionMode.CHORD && neuralRecognizerSelection == chordEngine
    }
    val loadFailedForCurrentSelection = neuralLoadFailedSelection == chordEngine
    val waitingForNeural = mode == DetectionMode.CHORD &&
        chordEngine != ChordEngine.CLASSIC &&
        selectedNeuralAssetsInstalled &&
        activeNeuralRecognizer == null &&
        !loadFailedForCurrentSelection

    if (granted && !waitingForNeural) {
        val engine = remember(
            profile,
            mode,
            chordEngine,
            referenceA4Hz,
            highPassCutoffHz,
            autoChordThreshold,
            chordMinScore,
            activeNeuralRecognizer,
        ) {
            val neural = activeNeuralRecognizer
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
                    currentOnResult(result)
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
