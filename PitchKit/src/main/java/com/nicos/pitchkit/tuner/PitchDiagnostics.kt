package com.nicos.pitchkit.tuner

/** Platform-neutral diagnostics. Enable explicitly in developer tools. */
object PitchDiagnostics {
    @Volatile var enabled = false
    var sink: (String, String) -> Unit = { tag, message -> System.err.println("$tag: $message") }
    fun d(tag: String, message: String) { if (enabled) sink(tag, message) }
    fun w(tag: String, message: String) { if (enabled) sink(tag, message) }
}
