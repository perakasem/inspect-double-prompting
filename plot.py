import matplotlib.pyplot as plt
from inspect_ai.analysis import evals_df

log_dir = "./logs/repetition_sweep"
df = evals_df(log_dir)

acc_col = [c for c in df.columns if "score" in c and "accuracy" in c]
if acc_col:
    df["accuracy"] = df[acc_col[0]] * 100
    df["accuracy_err"] = df["score_match_stderr"] * 100

df["reps"] = df["task_arg_num_reps"]
df["model_short"] = df["model"].astype(str).str.split("/").str[-1]

# Clean up names
model_name_map = {
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "gpt-4o": "GPT-4o",
    "gemini-2.0-flash-001": "Gemini 2.0 Flash",
}
task_name_map = {
    "gsm8k": "GSM8K",
    "arc_challenge": "ARC-Challenge",
    "math_eval": "MATH-500",
}

df["model_label"] = df["model_short"].map(model_name_map).fillna(df["model_short"])
df["task_label"] = df["task_name"].map(task_name_map).fillna(df["task_name"])

tasks = list(task_name_map.values())
models = list(model_name_map.values())
reps = sorted(df["reps"].unique())

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

colors = ["#4CAF93", "#F4845F", "#8FA8C8"]
rep_labels = [f"{r} rep{'s' if r > 1 else ''}" for r in reps]

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=False)

axes = axes.flatten()

x = range(len(models))
width = 0.22

def plot_task(ax, task_label):
    task_df = df[df["task_label"] == task_label]
    for i, rep in enumerate(reps):
        rep_df = task_df[task_df["reps"] == rep]
        accuracies, errors = [], []
        for m in models:
            row = rep_df[rep_df["model_label"] == m]
            accuracies.append(row["accuracy"].values[0] if len(row) else 0)
            errors.append(row["accuracy_err"].values[0] if len(row) else 0)
        offset = (i - len(reps) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x], accuracies, width,
            label=rep_labels[i], color=colors[i], alpha=0.9,
            yerr=errors, capsize=3, error_kw={"elinewidth": 1, "ecolor": "gray"}
        )

    ax.set_title(task_label, fontweight="bold", fontsize=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Accuracy (\\%)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 100)


for idx, task in enumerate(tasks):
    plot_task(axes[idx], task)

# Average across all tasks
ax = axes[len(tasks)]
for i, rep in enumerate(reps):
    rep_df = df[df["reps"] == rep].groupby("model_label")["accuracy"].mean()
    err_df = df[df["reps"] == rep].groupby("model_label")["accuracy_err"].mean()
    accuracies = [rep_df.get(m, 0) for m in models]
    errors = [err_df.get(m, 0) for m in models]
    offset = (i - len(reps) / 2 + 0.5) * width
    ax.bar(
        [xi + offset for xi in x], accuracies, width,
        label=rep_labels[i], color=colors[i], alpha=0.9,
        yerr=errors, capsize=3, error_kw={"elinewidth": 1, "ecolor": "gray"}
    )

ax.set_title("Average Across All Tasks", fontweight="bold", fontsize=12)
ax.set_xticks(list(x))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel("Accuracy (\\%)", fontsize=10)  # if using usetex
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis="y", linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()

plt.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.97, hspace=0.35, wspace=0.3)

legend = fig.legend(handles, labels, loc="lower center", ncol=3,
                    bbox_to_anchor=(0.5, 0.0), fontsize=11, title="Prompt Repetitions",
                    title_fontsize=11, frameon=True, fancybox=False, edgecolor="gray")


fig.suptitle("Effect of Prompt Repetition on Benchmark Accuracy",
             fontsize=15, fontweight="bold", y=0.97)

plt.savefig("repetition_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
