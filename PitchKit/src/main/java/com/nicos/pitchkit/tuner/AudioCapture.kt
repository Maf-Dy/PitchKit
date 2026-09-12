package com.nicos.pitchkit.tuner

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

internal class AudioCapture(
    val sampleRate: Int = 44100,
    val bufferSize: Int = 8192,
) {
    private var recorder: AudioRecord? = null
    private var job: Job? = null

    private val minBuffer = AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
    )

    @SuppressLint("MissingPermission")
    fun start(scope: CoroutineScope, onBuffer: (FloatArray) -> Unit) {
        val recordBuffer = maxOf(minBuffer, bufferSize * 2)
        recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            recordBuffer,
        )
        recorder?.startRecording()

        job = scope.launch(Dispatchers.IO) {
            val shorts = ShortArray(bufferSize)
            val floats = FloatArray(bufferSize)
            while (isActive) {
                val read = recorder?.read(shorts, 0, bufferSize) ?: 0
                if (read > 0) {
                    for (i in 0 until read) floats[i] = shorts[i] / 32768f
                    onBuffer(floats.copyOf(read))
                }
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
    }
}
