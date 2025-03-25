package com.example.fruitcam

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.fruitcam.ui.theme.FruitcamTheme
import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            CameraPreview()
        }
    }
}

private fun sendImageToServer() {
    Log.d("log", "hallo")
}

@OptIn(ExperimentalEncodingApi::class)
@SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
@Composable
fun CameraPreview() {
    val context = LocalContext.current
    val cameraViewModel = CameraPreviewViewModel()
    FruitcamTheme {
        Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
            CameraPreviewScreen(cameraViewModel)
            Column(
                Modifier
                    .fillMaxHeight()
                    .fillMaxWidth()
                    .padding(bottom = 50.dp),
                verticalArrangement = Arrangement.Bottom

            ) {
                var isProcessing by remember { mutableStateOf(false) }
                if (!isProcessing) {
                    Button(
                        onClick = {
                            isProcessing = true
                            cameraViewModel.captureImage(
                                context,
                                onImageCaptured = { byteArray ->
                                    Log.d("log", Base64.Default.encode(byteArray))
                                    Log.d("log", "${Base64.Default.decode(Base64.Default.encode(byteArray)).size}")
                                    Log.d("log", Base64.Default.encode(Base64.Default.decode(Base64.Default.encode(byteArray))))
                                    Log.d("log", "got bytes ${byteArray.size}")
                                    cameraViewModel.saveImageUsingMediaStore(context, byteArray, "testpic")
                                    isProcessing = false
                                })
                            sendImageToServer()
                        },
                        shape = CircleShape,
                        colors = ButtonDefaults.buttonColors(containerColor = Color.White),
                        modifier = Modifier
                            .size(width = 60.dp, height = 60.dp)
                            .align(Alignment.CenterHorizontally),
                    ) {}
                } else {
                    CircularProgressIndicator(
                        color = Color.White,
                        modifier = Modifier
                            .size(width = 60.dp, height = 60.dp)
                            .align(
                                Alignment.CenterHorizontally
                            ),
                    )
                }
            }
        }
    }
}
