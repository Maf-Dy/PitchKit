package com.nicos.pitchkit.tuner

import java.io.InputStream
import java.io.File

interface ModelAssets {
    fun list(directory: String): Array<String>?
    fun open(path: String): InputStream
}

class DirectoryModelAssets(private val root: File) : ModelAssets {
    private fun resolve(path: String): File {
        val file = File(root, path).canonicalFile
        require(file.toPath().startsWith(root.canonicalFile.toPath())) { "Invalid model asset path" }
        return file
    }
    override fun list(directory: String) = resolve(directory).list()
    override fun open(path: String) = resolve(path).inputStream()
}
