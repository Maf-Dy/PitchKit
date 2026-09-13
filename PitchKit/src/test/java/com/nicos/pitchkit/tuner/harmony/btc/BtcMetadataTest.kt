package com.nicos.pitchkit.tuner.harmony.btc

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class BtcMetadataTest {
    @Test
    fun parsesPinnedExportMetadata() {
        val metadata = BtcMetadata.parse(
            """
            mean=-2.5
            std=1.25
            model_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
            source_commit=aa6e3a8d7b017f082fd2aaff9329d5c26af49c03
            checkpoint_blob_sha=a691af56aec01fcd52d3bd46992f96bb9874ee59
            """.trimIndent().toByteArray()
        )

        assertEquals(-2.5f, metadata.mean)
        assertEquals(1.25f, metadata.std)
        assertEquals("aa6e3a8d7b017f082fd2aaff9329d5c26af49c03", metadata.sourceCommit)
        assertEquals("a691af56aec01fcd52d3bd46992f96bb9874ee59", metadata.checkpointBlobSha)
    }

    @Test
    fun rejectsNonPositiveStd() {
        assertThrows(IllegalArgumentException::class.java) {
            BtcMetadata.parse(
                """
                mean=0
                std=0
                model_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
                source_commit=x
                checkpoint_blob_sha=y
                """.trimIndent().toByteArray()
            )
        }
    }
}
