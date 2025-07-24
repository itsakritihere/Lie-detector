import pandas as pd
import matplotlib.pyplot as plt

# Load the MFCC data
df = pd.read_excel("features_lie4.xlsx")

# Determine the max time for consistent x-axis
total_time = df["Time (s)"].max()

# Create subplots (4 rows x 4 columns)
fig, axes = plt.subplots(4, 4, figsize=(18, 12))
fig.suptitle('Individual MFCC Coefficients Over Time', fontsize=18)

# Plot MFCC_1 to MFCC_13
for i in range(1, 14):
    row = (i - 1) // 4
    col = (i - 1) % 4
    ax = axes[row, col]
    ax.plot(df["Time (s)"], df[f"MFCC_{i}"], color='red')
    ax.set_title(f"MFCC_{i}", fontsize=11)
    ax.set_xlim(0, total_time)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.grid(True)

# Hide the last unused subplot
axes[3, 3].axis('off')  # 16th subplot not used

# Layout adjustment
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.show()
