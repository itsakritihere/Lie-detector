import librosa
import numpy as np
import pandas as pd

# Path to your audio file
file_path = "cleaning/cleaned_data/truth/t13.wav"
output_name = "t1"

# Frame and hop size for 10ms hop at 16kHz
frame_length = 320  # ~20 ms
hop_length = 160    # ~10 ms

def extract_framewise_features(file_path, output_name):
    y, sr = librosa.load(file_path, sr=None)

    # Extract raw features
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Pitch extraction using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=80,
        fmax=300,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # --- Corrections for Accuracy ---

    # Replace NaNs in pitch with minimum voiced frequency or a small default
    pitch_min_value = 80.0  # fmin
    f0_clean = np.where(np.isnan(f0), pitch_min_value, f0)

    # Avoid complete 0s in spectral_flux — replace zeros with a small value
    flux_clean = np.where(flux <= 1e-4, 1e-4, flux)

    # Avoid zeros in energy and centroid as well
    energy_clean = np.where(energy <= 1e-6, 1e-6, energy)
    centroid_clean = np.where(centroid <= 1e-6, 1e-6, centroid)
    zcr_clean = np.where(zcr <= 1e-6, 1e-6, zcr)

    # Align flux if needed
    if len(flux_clean) != len(zcr_clean):
        flux_clean = np.interp(np.linspace(0, len(flux_clean), num=len(zcr_clean)), np.arange(len(flux_clean)), flux_clean)

    # Stack features
    features = np.vstack([
        zcr_clean,
        energy_clean,
        f0_clean,
        centroid_clean,
        flux_clean
    ]).T

    # Time axis
    times = librosa.frames_to_time(np.arange(features.shape[0]), sr=sr, hop_length=hop_length)

    # Create DataFrame
    df = pd.DataFrame(features, columns=["ZCR", "Energy", "Pitch", "Spectral_Centroid", "Spectral_Flux"])
    df["Time (s)"] = times

    # Save
    output_file = f"{output_name}of13.xlsx"
    df.to_excel(output_file, index=False)
    print(f"✅ Saved enhanced features to '{output_file}'")

# Run it
extract_framewise_features(file_path, output_name)
