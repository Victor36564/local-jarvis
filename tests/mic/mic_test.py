import argparse
import os
import subprocess
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

# --- Configuration Constants ---
WAKE_WORD = "hey_jarvis"  # The wake word to listen for

# Initialize the openWakeWord model globally
oww_model = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")


# ==========================================
# 1. MICROPHONE METER TEST MODE (--mic)
# ==========================================

def mic_audio_callback(indata, frames, time, status):
    """Callback dedicated strictly to measuring microphone volume levels."""
    if status:
        print(status)
    
    # Calculate the Root Mean Square (RMS) to get the overall volume level
    volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
    meter_length = int(volume_norm * 10)
    meter = "█" * meter_length
    print(f"Volume: {meter:<50}", end="\r")

def test_mic():
    # Mic testing works perfectly at standard high-quality 44.1kHz
    MIC_SAMPLE_RATE = 44100  
    MIC_BLOCK_SIZE = 2048    

    print("🎙️ Starting continuous listening stream...")
    print("🔊 Speak into your mic to see the volume meter. Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(callback=mic_audio_callback,
                            channels=1,
                            samplerate=MIC_SAMPLE_RATE,
                            blocksize=MIC_BLOCK_SIZE):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\n🛑 Stream stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


# ==========================================
# 2. WAKE WORD TEST MODE (--wake)
# ==========================================

def trigger_action():
    """Action triggered natively on Windows when the wake word is heard."""
    print("\n💥 WAKE WORD DETECTED! Running action...")
    try:
        subprocess.Popen(["notepad.exe"])
    except Exception as e:
        print(f"Failed to open notepad: {e}")

def wake_audio_callback(indata, frames, time, status):
    """Callback dedicated strictly to streaming data into openWakeWord."""
    if status:
        print(status)
        
    # openWakeWord expects 16-bit PCM integer audio data, not floats
    audio_int16 = (indata * 32767).astype(np.int16).flatten()
    
    # Feed the frame into the model and fetch the wake word's probability dictionary
    predictions = oww_model.predict(audio_int16)
    
    # Check if confidence threshold exceeds 0.5
    if predictions[WAKE_WORD] > 0.5:
        trigger_action()
        oww_model.reset()

def test_wake():
    # CRITICAL: openWakeWord models are trained strictly on 16kHz audio!
    # They require processing chunks in specific buffer sizes (1280 works best)
    WAKE_SAMPLE_RATE = 16000  
    WAKE_BLOCK_SIZE = 1280    

    print(f"🎙️ Actively listening for the wake word: '{WAKE_WORD}'...")
    print("🚀 Say the wake word to trigger a Windows action! Press Ctrl+C to exit.\n")

    try:
        with sd.InputStream(callback=wake_audio_callback,
                            channels=1,
                            samplerate=WAKE_SAMPLE_RATE,
                            blocksize=WAKE_BLOCK_SIZE):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\n🛑 Voice assistant stopped.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


# ==========================================
# MAIN EXECUTION ROUTER
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test microphone input or wake word detection.")
    parser.add_argument("--mic", action="store_true", help="Test microphone input and volume meter.")
    parser.add_argument("--wake", action="store_true", help="Test wake word detection.")
    args = parser.parse_args()

    if args.wake:
        test_wake()  
    elif args.mic:
        test_mic()  
    else:
        # Fallback if the user types the script naked without flags
        parser.print_help()
