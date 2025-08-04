import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt



# === CONFIGURATION ===
AUDIO_PATH = "cleaning/cleaned_data/lie/lie10.wav"  # ← Update with your file path
SAMPLE_RATE = 16000
N_MFCC = 13
NFFT = 512
frame_size = 0.025  # 25 ms
frame_stride = 0.01  # 10 ms
nfilt = 40

# === Step 1: Load audio ===
y, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE)
print(f"Audio loaded: {AUDIO_PATH} | Sample rate: {sr} | Length: {len(y)/sr:.2f}s")

# === Step 2: Pre-emphasis filter ===
def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

y_preemphasized = pre_emphasis(y)

# === Step 2.5: Plot pre-emphasized signal ===
plt.figure(figsize=(12, 3))
plt.plot(y_preemphasized)
plt.title("Pre-emphasized Audio Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# === Step 3: Framing ===
frame_length, frame_step = frame_size * sr, frame_stride * sr
signal_length = len(y_preemphasized)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(y_preemphasized, z)

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
          np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

# === Step 3.5: Plot first frame ===
plt.figure(figsize=(14, 4))
plt.plot(frames[0])
plt.title('First Frame of the Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

# === Step 4: Apply Hamming window ===
frames *= np.hamming(frame_length)

# === Step 4.5: Plot windowed first frame ===
plt.figure(figsize=(14, 4))
plt.plot(frames[0])
plt.title('First Frame after Windowing')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

# === Step 5: FFT and Power Spectrum ===
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

plt.figure(figsize=(14, 4))
plt.plot(mag_frames[0])
plt.title('Magnitude Spectrum of the First Frame')
plt.xlabel('Frequency Bin')
plt.ylabel('Amplitude')
plt.show()

# === Step 6: Mel Filterbank ===
low_freq_mel = 0
high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
hz_points = 700 * (10**(mel_points / 2595) - 1)
bin = np.floor((NFFT + 1) * hz_points / sr)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])
    f_m = int(bin[m])
    f_m_plus = int(bin[m + 1])

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
filter_banks = 20 * np.log10(filter_banks)

plt.figure(figsize=(14, 5))
plt.imshow(filter_banks.T, cmap='hot', aspect='auto')
plt.title('Filter Bank Energies')
plt.xlabel('Frame Index')
plt.ylabel('Filter Index')
plt.show()
# === Step 7: MFCC Computation ===
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=NFFT, hop_length=int(sr * frame_stride))

# === Step 7.5: Plot MFCCs ===
plt.figure(figsize=(14, 5))
librosa.display.specshow(mfcc, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.xlabel('Time (s)')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.show()

# === Step 8: Tabular Feature Extraction per Frame ===
features = []
for i in range(mfcc.shape[1]):
    frame_features = {
        'Frame': i,
        'Time (s)': round(i * frame_stride, 3),
        'MFCC_1': round(mfcc[0, i], 3),
        'MFCC_2': round(mfcc[1, i], 3),
        'MFCC_3': round(mfcc[2, i], 3),
        'MFCC_4': round(mfcc[3, i], 3),
        'MFCC_5': round(mfcc[4, i], 3),
        'MFCC_6': round(mfcc[5, i], 3),
        'MFCC_7': round(mfcc[6, i], 3),
        'MFCC_8': round(mfcc[7, i], 3),
        'MFCC_9': round(mfcc[8, i], 3),
        'MFCC_10': round(mfcc[9, i], 3),
        'MFCC_11': round(mfcc[10, i], 3),
        'MFCC_12': round(mfcc[11, i], 3),
        'MFCC_13': round(mfcc[12, i], 3)
    }
    features.append(frame_features)

# === Step 9: Save to Excel ===
df = pd.DataFrame(features)
df.to_excel("fl10.xlsx", index=False)
print("Feature extraction completed. Saved as 'fl10.xlsx'.")

