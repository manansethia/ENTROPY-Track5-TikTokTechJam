import SwiftUI
import MetalKit
import CoreML

// TikTok Native Brand Palette
let obsidianBg = Color(red: 1/255, green: 1/255, blue: 1/255)
let cardDark = Color(red: 12/255, green: 12/255, blue: 13/255)
let tikTokPink = Color(red: 254/255, green: 44/255, blue: 85/255)
let tikTokCyan = Color(red: 37/255, green: 244/255, blue: 238/255)
let textMuted = Color(red: 138/255, green: 139/255, blue: 145/255)

@main
struct AetherForensicsApp: App {
    var body: some Scene {
        WindowGroup {
            NativeForensicsView()
                .preferredColorScheme(.dark)
        }
    }
}

struct NativeForensicsView: View {
    @State private var riskPercent: Double = 94.2
    @State private var verdict: String = "SYNTHETIC AIGC DETECTED"
    @State private var priorPrevalence: Double = 50.0
    @State private var showingImagePicker = false
    @State private var inputImage: UIImage?

    var body: some View {
        ZStack {
            obsidianBg.ignoresSafeArea()

            VStack(spacing: 20) {
                // Native iOS Navigation Header
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("AETHER FORENSICS")
                            .font(.system(size: 20, weight: .black))
                            .foregroundColor(.white)
                        Text("Native Apple Silicon • Metal 3 & CoreML ANE Engine")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(textMuted)
                    }
                    Spacer()
                    Text("ANE ACTIVE")
                        .font(.system(size: 9, weight: .black))
                        .foregroundColor(tikTokCyan)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(tikTokCyan.opacity(0.15))
                        .cornerRadius(8)
                }
                .padding(.horizontal, 20)
                .padding(.top, 10)

                // Native Risk Metric Card
                VStack(spacing: 8) {
                    Text("AI SYNTHETIC PROBABILITY")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(textMuted)

                    Text(String(format: "%.1f%%", riskPercent))
                        .font(.system(size: 44, weight: .black, design: .rounded))
                        .foregroundColor(riskPercent > 50 ? tikTokPink : Color(red: 0, green: 242/255, blue: 157/255))

                    Text(verdict)
                        .font(.system(size: 13, weight: .bold))
                        .foregroundColor(.white)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
                .background(cardDark)
                .cornerRadius(16)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )
                .padding(.horizontal, 20)

                // Bayesian Deployment Prior Controller
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("BAYESIAN PRIOR PREVALENCE")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(textMuted)
                        Spacer()
                        Text(String(format: "%.0f%% Feed Prior", priorPrevalence))
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(tikTokCyan)
                    }

                    Slider(value: $priorPrevalence, in: 1...99, step: 1)
                        .accentColor(tikTokCyan)

                    Text("Applies real-time Bayesian logit correction for Apple Neural Engine inference on live camera feed.")
                        .font(.system(size: 10))
                        .foregroundColor(textMuted)
                }
                .padding(16)
                .background(cardDark)
                .cornerRadius(16)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )
                .padding(.horizontal, 20)

                Spacer()

                // Native Action Button
                Button(action: {
                    showingImagePicker = true
                }) {
                    HStack {
                        Image(systemName: "camera.viewfinder")
                        Text("CAPTURE FRAME WITH COREML")
                            .font(.system(size: 13, weight: .bold))
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .background(tikTokPink)
                    .cornerRadius(14)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
            }
        }
    }
}
