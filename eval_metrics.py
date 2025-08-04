import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample performance data
metrics_data = {
    "Model": [
        "Naive Bayes", "Decision Tree", "Random Forest",
        "Logistic Regression", "SVM", "Voting Classifier"
    ],
    "Original_Accuracy": [0.83, 0.96, 0.96, 0.97, 0.97, 0.97],
    "Resampled_Accuracy": [0.76, 0.91, 0.97, 0.83, 0.89, 0.94],
    "Original_Precision": [0.82, 0.95, 0.95, 0.96, 0.94, 0.96],
    "Resampled_Precision": [0.78, 0.91, 0.97, 0.84, 0.89, 0.94],
    "Original_Recall": [0.88, 0.96, 0.96, 0.97, 0.97, 0.97],
    "Resampled_Recall": [0.73, 0.91, 0.97, 0.83, 0.89, 0.94],
    "Original_F1": [0.84, 0.96, 0.96, 0.96, 0.96, 0.96],
    "Resampled_F1": [0.76, 0.91, 0.97, 0.83, 0.89, 0.94],
}

df = pd.DataFrame(metrics_data)
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

# ============ LINE PLOTS ============
def plot_line(metric, ax):
    x = df["Model"]
    orig = df[f"Original_{metric}"]
    resamp = df[f"Resampled_{metric}"]

    ax.plot(x, orig, label="Original", marker='o', linewidth=2.5, color=palette[0])
    ax.plot(x, resamp, label="Resampled (SMOTE)", marker='s', linewidth=2.5, color=palette[1])
    ax.set_title(f"{metric} Comparison", fontsize=13, fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_ylim(0.70, 1.0)
    ax.tick_params(axis='x', rotation=30)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
plot_line("Accuracy", axes[0][0])
plot_line("Precision", axes[0][1])
plot_line("Recall", axes[1][0])
plot_line("F1", axes[1][1])
fig.suptitle("ML Model Performance Comparison (Line Graphs)", fontsize=16, fontweight="bold")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ml_line_graphs.png", dpi=300)
plt.show()

# ============ BAR PLOTS ============
def plot_bar(metric):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df["Model"]))
    width = 0.35

    ax.bar([i - width / 2 for i in x], df[f"Original_{metric}"], width=width, label='Original', color=palette[2])
    ax.bar([i + width / 2 for i in x], df[f"Resampled_{metric}"], width=width, label='Resampled (SMOTE)', color=palette[3])
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=30)
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison - Bar Chart", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"ml_bar_{metric.lower()}.png", dpi=300)
    plt.show()

# Generate bar charts for each metric
for m in ["Accuracy", "Precision", "Recall", "F1"]:
    plot_bar(m)
