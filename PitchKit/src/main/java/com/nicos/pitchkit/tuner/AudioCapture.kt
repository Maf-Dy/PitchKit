package com.nicos.pitchkit.tuner
// Modified in Maf-Dy/PitchKit fork: DSP correctness, performance, and lifecycle fixes.

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
    val sampleRate: Int = 44_100,
    val bufferSize: Int = 8192,
) {
    private var recorder: AudioRecord? = null
    private var job: Job? = null

    private val minBufferSize = AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
    )

    @SuppressLint("MissingPermission")
    fun start(scope: CoroutineScope, onBuffer: (FloatArray) -> Unit) {
        stop()
        val recordBuffer = maxOf(minBufferSize, bufferSize * 2)
        val audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            recordBuffer,
        )
        if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            audioRecord.release()
            return
        }

        recorder = audioRecord
        audioRecord.startRecording()
        job = scope.launch(Dispatchers.IO) {
            val shorts = ShortArray(bufferSize)
            val floats = FloatArray(bufferSize)
            while (isActive) {
                var filled = 0
                while (filled < bufferSize && isActive) {
                    val read = audioRecord.read(
                        shorts,
                        filled,
                        bufferSize - filled,
                        AudioRecord.READ_BLOCKING,
                    )
                    if (read <= 0) {
                        filled = 0
                        break
                    }
                    filled += read
                }
                if (filled != bufferSize) continue

                var i = 0
                while (i < bufferSize) {
                    floats[i] = shorts[i] / 32768f
                    i++
                }
                onBuffer(floats)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
        val activeRecorder = recorder ?: return
        recorder = null
        runCatching { activeRecorder.stop() }
        activeRecorder.release()
    }
}
