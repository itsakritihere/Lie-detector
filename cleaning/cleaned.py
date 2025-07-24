import os
import librosa
import numpy as np
from scipy.io.wavfile import write
import noisereduce as nr

# === CONFIG ===
INPUT_DIR = "filtered_data"              # ← works on filtered data
OUTPUT_DIR = "cleaned_data"            # ← saves trimmed + padded files
SAMPLE_RATE = 16000
# TARGET_DURATION = 15.0  # in seconds
# TARGET_LENGTH = int(SAMPLE_RATE * TARGET_DURATION)

# === Ensure output folders exist ===
os.makedirs(os.path.join(OUTPUT_DIR, "lie"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "truth"), exist_ok=True)

# === Trim, Denoise, Normalize, Pad ===
def trim_and_pad(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Step 1: Noise Reduction
    y = nr.reduce_noise(y=y, sr=sr)

    # Step 2: Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    # Step 3: Normalize amplitude
    if np.max(np.abs(y_trimmed)) > 0:
        y_trimmed = y_trimmed / np.max(np.abs(y_trimmed))

    # Step 4: Pad or truncate to fixed length
    # if len(y_trimmed) < TARGET_LENGTH:
    #     y_padded = np.pad(y_trimmed, (0, TARGET_LENGTH - len(y_trimmed)), mode='constant')
    # else:
    #     y_padded = y_trimmed[:TARGET_LENGTH]

    # Step 5: Clipping protection
    y_trimmed = np.clip(y_trimmed, -1.0, 1.0)

    return y_trimmed

# === Process All Files in filtered/ ===
for label in ["lie", "truth"]:
    input_folder = os.path.join(INPUT_DIR, label)
    output_folder = os.path.join(OUTPUT_DIR, label)

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            processed_audio = trim_and_pad(input_path)
            write(output_path, SAMPLE_RATE, processed_audio.astype(np.float32))
            print(f"Processed {file} → {output_path}")
