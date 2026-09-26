plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.compose)
    id("maven-publish")
}

android {
    namespace = "com.nicos.pitchkit"
    compileSdk = 37

    defaultConfig {
        minSdk = 28
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }

    // The live replay harness (src/test/.../tuner/live) constructs the real
    // streaming recognizers on the JVM. They log through android.util.Log under
    // BuildConfig.DEBUG, which testDebugUnitTest sets true.
    testOptions {
        unitTests {
            isReturnDefaultValues = true
        }
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.onnxruntime.android)

    testImplementation(libs.junit)
    // Real org.json for JVM unit tests; the Android stub throws on every call.
    testImplementation(libs.json)
    // Desktop ONNX Runtime so the live replay harness can run the shipped models
    // on the JVM. Same ai.onnxruntime API as the Android artifact.
    testImplementation(libs.onnxruntime.jvm)
}

publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "com.github.Maf-Dy"
            artifactId = "PitchKit"
            version = "1.1.0-modern"
            afterEvaluate {
                from(components["release"])
            }
        }
    }
}
