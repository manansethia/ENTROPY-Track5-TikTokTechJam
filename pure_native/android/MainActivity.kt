package com.tiktok.aetherforensics

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// TikTok Native Brand Palette
val ObsidianBg = Color(0xFF010101)
val CardDark = Color(0xFF0C0C0D)
val TikTokPink = Color(0xFFFE2C55)
val TikTokCyan = Color(0xFF25F4EE)
val TextMuted = Color(0xFF8A8B91)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AetherForensicsNativeScreen()
        }
    }
}

@Composable
fun AetherForensicsNativeScreen() {
    var riskPercent by remember { mutableStateOf(94.2f) }
    var verdict by remember { mutableStateOf("SYNTHETIC AIGC DETECTED") }
    var priorPrevalence by remember { mutableStateOf(50f) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ObsidianBg)
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Native Header
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 24.dp, bottom = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("AETHER FORENSICS", color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.Black)
                Text("Native Android Kotlin Engine • Qualcomm / ARM64", color = TextMuted, fontSize = 11.sp)
            }
            Surface(
                color = Color(0x3325F4EE),
                shape = RoundedCornerShape(12.dp)
            ) {
                Text("NNAPI GPU", color = TikTokCyan, fontSize = 10.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Verdict Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = CardDark)
        ) {
            Column(modifier = Modifier.padding(20.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text("AI SYNTHETIC PROBABILITY", color = TextMuted, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "${"%.1f".format(riskPercent)}%",
                    color = if (riskPercent > 50f) TikTokPink else Color(0xFF00F29D),
                    fontSize = 42.sp,
                    fontWeight = FontWeight.Black
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = verdict,
                    color = Color.White,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // Bayesian Prior Slider
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = CardDark)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("BAYESIAN PRIOR SENSITIVITY", color = TextMuted, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "${priorPrevalence.toInt()}% Feed Prevalence",
                    color = TikTokCyan,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold
                )
                Slider(
                    value = priorPrevalence,
                    onValueChange = { priorPrevalence = it },
                    valueRange = 1f..99f,
                    colors = SliderDefaults.colors(
                        thumbColor = Color.White,
                        activeTrackColor = TikTokCyan
                    )
                )
                Text(
                    "Calibrates on-device decision boundary for social media stream distributions.",
                    color = TextMuted,
                    fontSize = 10.sp
                )
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        // Action Button
        Button(
            onClick = { /* Native Android Camera / Gallery Pick */ },
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.buttonColors(containerColor = TikTokPink)
        ) {
            Text("SCAN FRAME WITH ON-DEVICE ANE", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        }
    }
}
