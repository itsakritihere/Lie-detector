import pandas as pd
import numpy as np
from scipy.stats import entropy, skew
from fuzzy import predict_truth_or_lie_from_features  # You must define this externally

def extract_refined_features_from_excel(mfcc_file, other_file):
    df_mfcc = pd.read_excel(mfcc_file)
    df_other = pd.read_excel(other_file)

    df_mfcc = df_mfcc.select_dtypes(include=[np.number]).dropna()
    df_other = df_other.select_dtypes(include=[np.number]).dropna()
    df = pd.concat([df_mfcc, df_other], axis=1)

    # Extract individual features
    zcr = df['ZCR'].values
    energy = df['Energy'].values
    pitch = df['Pitch'].values
    flux = df['Spectral_Flux'].values
    centroid = df['Spectral_Centroid'].values

    mfcc_columns = [df[f'MFCC_{i}'].values for i in range(1, 14)]
    mfcc13 = mfcc_columns[-1]

    frame_duration_sec = 0.01
    duration = len(zcr) * frame_duration_sec

    # Pause features
    pause_count = np.sum((zcr < 0.05) & (energy < 0.01))
    silent = ((zcr < 0.02) & (energy < 0.005)).astype(int)

    longest_pause = 0
    count = 0
    for val in silent:
        if val == 1:
            count += 1
            longest_pause = max(longest_pause, count)
        else:
            count = 0
    longest_pause *= frame_duration_sec

    pause_rate = pause_count / duration if duration > 0 else 0

    # Clean pitch
    pitch = pitch[~np.isnan(pitch)]
    if len(pitch) < 2:
        pitch_jump_count = 0
        pitch_range = 0
        pitch_skewness = 0
        pitch_std = 0
    else:
        # Adaptive pitch analysis
        valid_pitch = pitch[pitch > 100]
        if len(valid_pitch) > 0:
            pitch_range = np.max(valid_pitch) - np.min(valid_pitch)
            pitch_skewness = skew(valid_pitch)
        else:
            pitch_range = 0
            pitch_skewness = 0

        pitch_diff = np.abs(np.diff(pitch))
        jump_threshold = np.percentile(pitch_diff, 90)
        pitch_jump_count = np.sum(pitch_diff > jump_threshold)
        pitch_std = np.std(pitch)

    # MFCC microstress: mean of all MFCC std deviations
    mfcc_microstress = np.mean([np.std(mfcc) for mfcc in mfcc_columns])

    # Spectral flux spikes
    flux_spike_threshold = np.percentile(flux, 75)
    flux_spikes = np.sum(flux > flux_spike_threshold)

    # Energy range
    energy_range = np.max(energy) - np.min(energy)

    # Pitch entropy
    if pitch is None or len(pitch) == 0 or np.isnan(pitch).all():
        pitch_entropy = 0.0
    else:
        clean_pitch = pitch[~np.isnan(pitch)]
        if len(clean_pitch) == 0:
            pitch_entropy = 0.0
        else:
            hist, _ = np.histogram(clean_pitch, bins=20, density=True)
            pitch_entropy = entropy(hist + 1e-6)

    # Spectral rolloff (95th percentile)
    spectral_rolloff = np.percentile(centroid, 95)

    # Stress index (normalized)
    norm_pitch_skew = pitch_skewness / (pitch_range + 1e-6) if pitch_range > 0 else 0
    norm_flux_spike = flux_spikes / len(flux) if len(flux) > 0 else 0
    norm_mfcc_stress = mfcc_microstress / (np.max(mfcc13) + 1e-6)

   

    return [
        pause_count, longest_pause, pause_rate, pitch_jump_count,
        pitch_range, pitch_skewness, pitch_std, mfcc_microstress, flux_spikes,
        energy_range, pitch_entropy, spectral_rolloff
    ]

refined_feature_names = [
    "pause_count", "longest_pause", "pause_rate", "pitch_jump_count",
    "pitch_range", "pitch_skewness", "pitch_std", "mfcc13_microstress", "flux_spikes",
    "energy_range", "pitch_entropy", "spectral_rolloff"
]

if __name__ == "__main__":
    mfcc_file = "fl15.xlsx"
    other_file = "l1l1offl15.xlsx"

    features = extract_refined_features_from_excel(mfcc_file, other_file)

    print(f"\nBehavioral Feature Analysis for:")
    print(f"MFCC file : {mfcc_file}")
    print(f"Other file: {other_file}\n")

    for name, value in zip(refined_feature_names, features):
        print(f"{name:25s}: {value:.4f}")

result, score = predict_truth_or_lie_from_features(features)

print(f"\nFinal Fuzzy Logic Prediction: {result.upper()}, Score: {score}")

# Save to CSV with prediction only
df = pd.DataFrame([features + [result]], columns=refined_feature_names + ["prediction"])
df.to_csv("feature_data.csv", index=False)
print("Features + prediction saved to 'feature_data.csv'")