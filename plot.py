import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend that doesn't require tkinter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read results
results = pd.read_csv("caffe_results_baseline.csv")
results_rf = pd.read_csv("caffe_results_v3.csv")

# extract values
labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
baseline = results[labels].iloc[0].values
improved = results_rf[labels].iloc[0].values

# setup bar chart
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width / 2, baseline, width, label="Naive Bayes Baseline", color="red")
bars2 = ax.bar(x + width / 2, improved, width, label="Random Forest (v3)", color="purple")

# annotate bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

# final formatting
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Naive Bayes vs Random Forest (TF-IDF)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("caffe_nb_vs_rf_comparison.png")
print("Plot saved to caffe_nb_vs_rf_comparison.png")