package com.nicos.pitchkit.tuner.harmony.lvchordia

/** Kotlin's primitive arrays do not provide Iterable.mapIndexedNotNull. */
internal inline fun <R : Any> IntArray.mapIndexedNotNull(
    transform: (index: Int, value: Int) -> R?,
): List<R> {
    val result = ArrayList<R>(size)
    for (index in indices) {
        transform(index, this[index])?.let(result::add)
    }
    return result
}
