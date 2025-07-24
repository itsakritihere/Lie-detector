import pandas as pd
import matplotlib.pyplot as plt

# Load your Excel file here
excel_file = "l2_featurest2_10ms.xlsx"  # change if filename is different
df = pd.read_excel(excel_file)

# Features to plot
features = ["ZCR", "Energy", "Pitch", "Spectral_Centroid", "Spectral_Flux"]
time = df["Time (s)"]

# Plot setup
plt.figure(figsize=(15, 12))

for i, feature in enumerate(features, 1):
    plt.subplot(len(features), 1, i)
    plt.plot(time, df[feature], label=feature)
    plt.title(f"{feature} over Time")
    plt.xlabel("Time (s)")
    plt.ylabel(feature)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.suptitle("Audio Feature Graphs (10ms Frames)", fontsize=12, y=1)
plt.show()
