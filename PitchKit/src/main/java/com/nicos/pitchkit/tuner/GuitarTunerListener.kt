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
import com.nicos.pitchkit.tuner.harmony.btc.BtcAndroidFactory
import com.nicos.pitchkit.tuner.harmony.btc.BtcContract
import com.nicos.pitchkit.tuner.harmony.btc.BtcStreamingRecognizer
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetAndroidFactory
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetContract
import com.nicos.pitchkit.tuner.harmony.chordnet.ChordNetStreamingRecognizer
import com.nicos.pitchkit.tuner.harmony.crema.CremaAndroidFactory
import com.nicos.pitchkit.tuner.harmony.crema.CremaContract
import com.nicos.pitchkit.tuner.harmony.crema.CremaStreamingRecognizer
import com.nicos.pitchkit.tuner.harmony.solitito.SolititoAndroidFactory
import com.nicos.pitchkit.tuner.harmony.solitito.SolititoContract
import com.nicos.pitchkit.tuner.harmony.solitito.SolititoStreamingRecognizer
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
    val btcAssetsInstalled = remember(applicationContext) {
        BtcAndroidFactory.liveAssetsInstalled(applicationContext)
    }
    val solititoAssetsInstalled = remember(applicationContext) {
        SolititoAndroidFactory.assetsInstalled(applicationContext)
    }
    val selectedNeuralAssetsInstalled = when (chordEngine) {
        ChordEngine.AUTO -> cremaAssetsInstalled || chordNetAssetsInstalled
        ChordEngine.CREMA -> cremaAssetsInstalled
        ChordEngine.CHORD_NET -> chordNetAssetsInstalled
        ChordEngine.SOLITITO -> solititoAssetsInstalled
        ChordEngine.BTC_EXPERIMENTAL -> btcAssetsInstalled
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
        btcAssetsInstalled,
        solititoAssetsInstalled,
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

            fun trySolitito(): ChordRecognizer? {
                if (!solititoAssetsInstalled) {
                    if (BuildConfig.DEBUG) Log.d("PitchKit", "solitito assets are not installed")
                    return null
                }
                return try {
                    // No referenceA4Hz: solitito's kernel and root head are fixed at the
                    // shipped 440 Hz grid; see SolititoAndroidFactory.
                    SolititoAndroidFactory.create(context = applicationContext).also {
                        if (BuildConfig.DEBUG) Log.d("PitchKit", "solitito-ai recognizer loaded")
                    }
                } catch (error: Throwable) {
                    Log.e("PitchKit", "solitito-ai failed to load", error)
                    null
                }
            }

            fun tryBtc(): ChordRecognizer? {
                if (!btcAssetsInstalled) {
                    if (BuildConfig.DEBUG) Log.d("PitchKit", "BTC live assets are not installed")
                    return null
                }
                return try {
                    BtcAndroidFactory.createLiveRecognizer(
                        context = applicationContext,
                        referenceA4Hz = referenceA4Hz,
                    ).also {
                        if (BuildConfig.DEBUG) Log.d("PitchKit", "BTC experimental live recognizer loaded")
                    }
                } catch (error: Throwable) {
                    Log.e("PitchKit", "BTC experimental live failed to load", error)
                    null
                }
            }

            fun tryNeural(candidate: ChordEngine): ChordRecognizer? = when (candidate) {
                ChordEngine.CREMA -> tryCrema()
                ChordEngine.CHORD_NET -> tryChordNet()
                ChordEngine.SOLITITO -> trySolitito()
                ChordEngine.BTC_EXPERIMENTAL -> tryBtc()
                ChordEngine.AUTO, ChordEngine.CLASSIC -> null
            }

            when (requestedSelection) {
                // Crema first, ChordNet as the fallback: see ChordEngine.AUTO_PREFERENCE.
                ChordEngine.AUTO -> ChordEngine.AUTO_PREFERENCE.firstNotNullOfOrNull(::tryNeural)
                ChordEngine.CREMA -> tryCrema()
                ChordEngine.CHORD_NET -> tryChordNet()
                ChordEngine.SOLITITO -> trySolitito()
                ChordEngine.BTC_EXPERIMENTAL -> tryBtc()
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
                is BtcStreamingRecognizer -> BtcContract.SAMPLE_RATE
                is SolititoStreamingRecognizer -> SolititoContract.SAMPLE_RATE
                else -> 44100
            }
            val engineBufferSize = when (neural) {
                is CremaStreamingRecognizer -> CremaContract.HOP_LENGTH
                is ChordNetStreamingRecognizer -> ChordNetContract.HOP_LENGTH
                is BtcStreamingRecognizer -> BtcContract.HOP_LENGTH
                // 256 samples at 16 kHz is exactly one analysis hop and a 16 ms frame
                // period - the smallest input-latency term of any lane in the APK
                // (2 x 256 / 16000 = 32 ms).
                //
                // This is the *read* size AudioCapture delivers, not AudioRecord's own
                // ring: `AudioCapture` opens the recorder with
                // `maxOf(AudioRecord.getMinBufferSize(...), bufferSize * 2)` and still
                // hands the engine exactly `bufferSize` samples a time, so a device with
                // a large minimum costs latency inside the driver but does not coarsen
                // this cadence. If the 16 kHz negotiation fails and AudioCapture settles
                // on 44.1 or 48 kHz, the buffer stays 256 source samples (a *shorter*
                // 5.8 ms) and SolititoStreamingRecognizer's own resampler accumulates
                // them, so analysis frames still emerge every 256 samples at 16 kHz and
                // the 40 ms inference cadence is unchanged either way.
                is SolititoStreamingRecognizer -> SolititoContract.LIVE_BUFFER_SAMPLES
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
                // The neural lanes ask for UNPROCESSED first where the device
                // advertises it and fall back to VOICE_RECOGNITION, then MIC,
                // inside AudioCapture's own negotiation. Classic DSP is unchanged.
                preferredAudioSource = if (neural != null) {
                    preferredNeuralAudioSource(applicationContext)
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
