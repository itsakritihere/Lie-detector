import os
import librosa
import matplotlib.pyplot as plt

# === CONFIG ===
INPUT_DIR = "cleaning/cleaned_data"  # folder with 'lie' and 'truth'
OUTPUT_DIR = "waveforms"      # where to save plots
SAMPLE_RATE = 16000           # match your preprocessed sample rate

# Create output folders
os.makedirs(os.path.join(OUTPUT_DIR, "lie"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "truth"), exist_ok=True)

# === Function to plot waveform ===
def plot_waveform(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    plt.figure(figsize=(8, 3))
    plt.plot(y, color='blue')
    plt.title("Amplitude vs Samples")
    plt.xlabel("Number of Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# === Generate waveform plots ===
for label in ["lie", "truth"]:
    input_folder = os.path.join(INPUT_DIR, label)
    output_folder = os.path.join(OUTPUT_DIR, label)

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace(".wav", ".png"))
            plot_waveform(input_path, output_path)
            print(f"Saved waveform for {file} → {output_path}")
