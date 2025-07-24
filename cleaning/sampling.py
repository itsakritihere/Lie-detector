import os
import librosa
import matplotlib.pyplot as plt

# Set path to your folders
base_path = "normalized_data/"
folders = ["lie", "truth"]

sampling_rates = []

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue

    print(f"\n📁 Checking folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            print(f"🔍 Reading: {file_path}")
            try:
                _, sr = librosa.load(file_path, sr=None)
                sampling_rates.append(sr)
            except Exception as e:
                print(f"⚠️ Could not load {file_path}: {e}")

# Summary
print(f"\n✅ Total audio files checked: {len(sampling_rates)}")
print(f"📊 Unique sampling rates found: {set(sampling_rates)}")

# Plot
if sampling_rates:
    plt.figure(figsize=(6, 4))
    plt.hist(sampling_rates, bins=10, color='skyblue', edgecolor='black')
    plt.title("Sampling Rate Distribution")
    plt.xlabel("Sampling Rate (Hz)")
    plt.ylabel("Number of Files")
    plt.xticks(sorted(set(sampling_rates)))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No audio files processed.")
