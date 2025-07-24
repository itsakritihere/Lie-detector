from pydub import AudioSegment
import os

# ✅ Set full paths manually
AudioSegment.converter = "C:/Users/Anukriti Chauhan/Downloads/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/Users/Anukriti Chauhan/Downloads/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffprobe.exe"
input_path = "../data/"  # go up one level to reach 'data' from 'cleaning'

output_path = "normalized_data/"

os.makedirs(output_path + "lie", exist_ok=True)
os.makedirs(output_path + "truth", exist_ok=True)

for label in ["lie", "truth"]:
    for file in os.listdir(input_path + label):
        if file.endswith(".mp3") or file.endswith(".wav"):
            sound = AudioSegment.from_file(input_path + label + "/" + file)
            sound = sound.set_channels(1).set_frame_rate(16000)
            output_file = output_path + label + "/" + file.split('.')[0] + ".wav"
            sound.export(output_file, format="wav")
