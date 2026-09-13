package com.nicos.pitchkit.tuner

import android.annotation.SuppressLint
import android.media.AudioFormat
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

    private val poolLock = Any()
    private val floatBufferPool = ArrayDeque<FloatArray>()

    @SuppressLint("MissingPermission")
    fun start(
        scope: CoroutineScope,
        onBuffer: (samples: FloatArray, sampleRate: Int) -> Unit,
    ) {
        check(recorder == null) { "AudioCapture is already running" }
        resetPool()

        val sources = listOf(
            preferredAudioSource,
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            MediaRecorder.AudioSource.MIC,
        ).distinct()
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
            "AudioRecord started source=${sourceName(activeAudioSource)}($activeAudioSource) rate=$activeSampleRate buffer=$bufferSize",
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
                            "AudioRecord.read returned $read from ${sourceName(activeAudioSource)} at $activeSampleRate Hz (failure #$consecutiveFailures)",
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

    private fun sourceName(source: Int): String = when (source) {
        MediaRecorder.AudioSource.MIC -> "MIC"
        MediaRecorder.AudioSource.VOICE_RECOGNITION -> "VOICE_RECOGNITION"
        MediaRecorder.AudioSource.UNPROCESSED -> "UNPROCESSED"
        else -> "SOURCE"
    }
}
