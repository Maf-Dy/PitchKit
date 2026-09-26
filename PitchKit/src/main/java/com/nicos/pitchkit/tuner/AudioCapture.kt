package com.nicos.pitchkit.tuner

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

internal class AudioCapture(
    private val preferredSampleRate: Int = 44100,
    private val bufferSize: Int = 8192,
    private val preferredAudioSource: Int = MediaRecorder.AudioSource.MIC,
) {
    private companion object {
        const val BUFFER_POOL_SIZE = 3
    }

    private var recorder: AudioRecord? = null
    private var job: Job? = null
    private var activeSampleRate: Int = preferredSampleRate
    private var activeAudioSource: Int = preferredAudioSource

    /**
     * The audio source the negotiation actually opened, lower-cased for display,
     * or `null` before [start] has picked one. The caller cannot assume the
     * preferred source was granted — the rate loop is the outer loop, so a device
     * that cannot open the preferred rate re-negotiates the source too.
     */
    val activeSourceLabel: String?
        get() = if (recorder != null) audioSourceLabel(activeAudioSource) else null

    private val poolLock = Any()
    private val floatBufferPool = ArrayDeque<FloatArray>()

    @SuppressLint("MissingPermission")
    fun start(
        scope: CoroutineScope,
        onBuffer: (samples: FloatArray, sampleRate: Int) -> Unit,
    ) {
        check(recorder == null) { "AudioCapture is already running" }
        resetPool()

        val sources = audioSourceCandidates(preferredAudioSource)
        val rates = listOf(preferredSampleRate, 44100, 48000).distinct()

        var selected: AudioRecord? = null
        var selectedRate = preferredSampleRate
        var selectedSource = preferredAudioSource

        outer@ for (rate in rates) {
            val minimumBuffer = AudioRecord.getMinBufferSize(
                rate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            if (minimumBuffer <= 0) continue

            for (source in sources) {
                val candidate = runCatching {
                    AudioRecord(
                        source,
                        rate,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        maxOf(minimumBuffer, bufferSize * 2),
                    )
                }.getOrNull() ?: continue

                if (candidate.state != AudioRecord.STATE_INITIALIZED) {
                    candidate.release()
                    continue
                }

                val started = runCatching {
                    candidate.startRecording()
                    candidate.recordingState == AudioRecord.RECORDSTATE_RECORDING
                }.getOrDefault(false)

                if (!started) {
                    candidate.release()
                    continue
                }

                selected = candidate
                selectedRate = rate
                selectedSource = source
                break@outer
            }
        }

        recorder = selected ?: error("No supported AudioRecord configuration was available")
        activeSampleRate = selectedRate
        activeAudioSource = selectedSource

        Log.i(
            "PitchKitAudio",
            "AudioRecord started source=${audioSourceLabel(activeAudioSource)}($activeAudioSource) " +
                "requested=${audioSourceLabel(preferredAudioSource)} rate=$activeSampleRate buffer=$bufferSize",
        )

        job = scope.launch(Dispatchers.IO) {
            val shorts = ShortArray(bufferSize)
            var consecutiveFailures = 0

            while (isActive) {
                val read = recorder?.read(shorts, 0, bufferSize) ?: 0
                if (read > 0) {
                    consecutiveFailures = 0
                    val pooled = acquireFloatBuffer() ?: continue
                    for (i in 0 until read) pooled[i] = shorts[i] / 32768f

                    val delivered = if (read == bufferSize) {
                        pooled
                    } else {
                        pooled.copyOf(read).also { recycle(pooled) }
                    }

                    try {
                        onBuffer(delivered, activeSampleRate)
                    } catch (error: Throwable) {
                        recycle(delivered)
                        throw error
                    }
                } else {
                    consecutiveFailures++
                    if (consecutiveFailures == 1 || consecutiveFailures % 20 == 0) {
                        Log.w(
                            "PitchKitAudio",
                            "AudioRecord.read returned $read from ${audioSourceLabel(activeAudioSource)} at $activeSampleRate Hz (failure #$consecutiveFailures)",
                        )
                    }
                }
            }
        }
    }

    fun recycle(buffer: FloatArray) {
        if (buffer.size != bufferSize) return
        synchronized(poolLock) {
            if (floatBufferPool.size < BUFFER_POOL_SIZE) {
                floatBufferPool.addLast(buffer)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
        val activeRecorder = recorder ?: return
        runCatching { activeRecorder.stop() }
        activeRecorder.release()
        recorder = null
        Log.i("PitchKitAudio", "AudioRecord stopped")
    }

    private fun resetPool() {
        synchronized(poolLock) {
            floatBufferPool.clear()
            repeat(BUFFER_POOL_SIZE) {
                floatBufferPool.addLast(FloatArray(bufferSize))
            }
        }
    }

    private fun acquireFloatBuffer(): FloatArray? = synchronized(poolLock) {
        if (floatBufferPool.isEmpty()) null else floatBufferPool.removeFirst()
    }
}

/**
 * Source negotiation order: the caller's preference first, then the two sources
 * that shipped before `UNPROCESSED` was ever requested.
 *
 * Keeping `VOICE_RECOGNITION` and `MIC` in the tail is what makes asking for
 * `UNPROCESSED` free: a device that cannot open it drops straight back to the
 * behaviour it had, with no crash path.
 */
internal fun audioSourceCandidates(preferredAudioSource: Int): List<Int> = listOf(
    preferredAudioSource,
    MediaRecorder.AudioSource.VOICE_RECOGNITION,
    MediaRecorder.AudioSource.MIC,
).distinct()

/**
 * The source the neural chord lanes ask for first.
 *
 * `VOICE_RECOGNITION` is a voice-tuned front end — often band-limited and
 * AGC-shaped — feeding a music model, which the live benchmark plan flags as an
 * untested design choice. `UNPROCESSED` is the raw alternative, but it is only
 * meaningful where the device advertises it, so the property is consulted first
 * and anything unexpected falls back to the shipped source.
 */
internal fun preferredNeuralAudioSource(context: Context): Int {
    val supported = runCatching {
        val audio = context.applicationContext.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
        audio?.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED)
    }.getOrNull()
    return if (supported.equals("true", ignoreCase = true)) {
        MediaRecorder.AudioSource.UNPROCESSED
    } else {
        MediaRecorder.AudioSource.VOICE_RECOGNITION
    }
}

/** Short display name for a [MediaRecorder.AudioSource], used in logs and the backend string. */
internal fun audioSourceLabel(source: Int): String = when (source) {
    MediaRecorder.AudioSource.MIC -> "mic"
    MediaRecorder.AudioSource.VOICE_RECOGNITION -> "voice-recognition"
    MediaRecorder.AudioSource.UNPROCESSED -> "unprocessed"
    else -> "source-$source"
}
