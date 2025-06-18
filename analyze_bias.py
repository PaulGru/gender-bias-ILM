import os
import re
import pandas as pd
import matplotlib.pyplot as plt

bias_dir = "bias_outputs_2500" #_2500
plot_path = "bias_summary_plot.png"

# === CHARGER TOUS LES FICHIERS CSV ===
records = []
pattern = re.compile(r"split(\d+)_p(\d+)_lr([\d.eE+-]+)_steps(\d+)_(eLM|iLM)\.csv")

for fname in os.listdir(bias_dir):
    match = pattern.match(fname)
    seed, pval, lr_str, steps, method = match.groups()
    
    lr = float(lr_str)
    p_env_a = float("0." + pval[1:])
    steps = int(steps)
    
    df = pd.read_csv(os.path.join(bias_dir, fname))
    for val in df["bias"]:
        records.append({
            "seed": int(seed),
            "p_env_a": p_env_a,
            "lr": lr,
            "steps": steps,
            "method": method,
            "bias": val
        })

df_all = pd.DataFrame(records)
df_all = df_all[df_all['p_env_a'] == 0.9]
grouped = df_all.groupby(["steps", "lr", "method"])["bias"].mean().reset_index()


plt.figure(figsize=(10, 6))
markers = {"eLM": "o", "iLM": "s"}
linestyles = {"eLM": "-", "iLM": "--"}
colors = {
    ("eLM", 1e-5): "tab:blue",
    ("iLM", 1e-5): "tab:orange",
    ("eLM", 5e-5): "tab:green",
    ("iLM", 5e-5): "tab:red"
}

for (lr, method), sub in grouped.groupby(["lr", "method"]):
    color = colors.get((method, lr), None)
    plt.errorbar(
        sub["steps"], sub["bias"],
        label=f"{method} (lr={lr:.0e})",
        marker=markers[method], linestyle=linestyles[method],
        color=color, linewidth=2
    )

plt.title("Biais moyen ± écart-type selon le nombre de steps")
plt.xlabel("nombre de steps")
plt.ylabel("Biais moyen (1 - entropie normalisée)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)
plt.show()
