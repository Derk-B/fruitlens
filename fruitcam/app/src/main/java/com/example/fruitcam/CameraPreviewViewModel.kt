package com.example.fruitcam

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA
import androidx.camera.core.CameraSelector.DEFAULT_FRONT_CAMERA
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceRequest
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.compose.runtime.remember
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.nio.ByteBuffer

class CameraPreviewViewModel : ViewModel() {
    private val _resultText = MutableStateFlow("Initial Text")
    val resultText: StateFlow<String> = _resultText

    fun updateResultText(newText: String) {
        _resultText.value = newText
    }

    // Used to set up a link between the Camera and the UI.
    private val _surfaceRequest = MutableStateFlow<SurfaceRequest?>(null)
    val surfaceRequest: StateFlow<SurfaceRequest?> = _surfaceRequest
    val imageCapture: ImageCapture = ImageCapture.Builder().build()

    private val cameraPreviewUseCase = Preview.Builder().build().apply {
        setSurfaceProvider { newSurfaceRequest ->
            _surfaceRequest.update { newSurfaceRequest }
        }
    }

    suspend fun bindToCamera(appContext: Context, lifecycleOwner: LifecycleOwner) {
        val processCameraProvider = ProcessCameraProvider.awaitInstance(appContext)
        processCameraProvider.bindToLifecycle(
            lifecycleOwner, DEFAULT_BACK_CAMERA, cameraPreviewUseCase, imageCapture
        )

        // Cancellation signals we're done with the camera
        try {
            awaitCancellation()
        } finally {
            processCameraProvider.unbindAll()
        }
    }

    fun captureImage(appContext: Context, onImageCaptured: (ByteArray) -> Unit) {
        viewModelScope.launch {
            imageCapture.takePicture(
                ContextCompat.getMainExecutor(appContext),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        super.onCaptureSuccess(image)

                        val pixelData = imageProxyToByteArray(image)
                        image.close()

                        onImageCaptured(pixelData)
                    }

                    override fun onError(exception: ImageCaptureException) {
                        super.onError(exception)
                        updateResultText("Error capturing image: ${exception.message}")
                    }
                }
            )
        }
    }

    private fun imageProxyToByteArray(image: ImageProxy): ByteArray {
        val planeProxy = image.planes[0]
        val buffer: ByteBuffer = planeProxy.buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return bytes
    }
}