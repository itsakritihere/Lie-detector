import os
import librosa
import shutil

MAX_DURATION = 15.0  # seconds
input_dir = "normalized_data"
output_dir = "filtered_data"

os.makedirs(output_dir + "/lie", exist_ok=True)
os.makedirs(output_dir + "/truth", exist_ok=True)

for label in ["lie", "truth"]:
    path = os.path.join(input_dir, label)
    for file in os.listdir(path):
        if file.endswith(".wav"):
            file_path = os.path.join(path, file)
            try:
                duration = librosa.get_duration(path=file_path)
                if duration <= MAX_DURATION:
                    # Copy file to new folder if it's within limit
                    shutil.copy(file_path, os.path.join(output_dir, label, file))
                    print(f"✅ Kept {file} - {duration:.2f}s")
                else:
                    print(f"❌ Skipped {file} - {duration:.2f}s")
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
