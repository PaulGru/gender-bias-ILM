import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

means_dir       = "bias_outputs_xxx_mean"
plot_output_dir = "plots_par_r"
os.makedirs(plot_output_dir, exist_ok=True)

# 1. Agrégation des moyennes par r et par (r, lr)
aggregated_by_r        = defaultdict(list)  # clé = (r, modèle)
aggregated_by_r_and_lr = defaultdict(list)  # clé = (r, modèle, lr)

for filename in os.listdir(means_dir):
    # on ne garde que les CSV
    if not filename.lower().endswith(".csv"):
        continue

    filepath = os.path.join(means_dir, filename)
    # lecture du CSV (une seule ligne, colonne 'bias')
    df = pd.read_csv(filepath)
    if 'bias' in df.columns:
        mean_val = df['bias'].iloc[0]
    else:
        mean_val = df.iloc[0, 0]

    parts     = filename.rstrip(".csv").split("_")
    # extraire r (ex: "r10" -> 10)
    r_part    = next((p for p in parts if p.startswith("r")), None)
    # extraire lr (ex: "lr5e-05")
    lr_part   = next((p for p in parts if p.startswith("lr")), None)
    model_type= "eLM" if "eLM" in filename else "iLM"

    if r_part:
        try:
            r_value = int(r_part.replace("r", ""))
        except ValueError:
            continue
        aggregated_by_r[(r_value, model_type)].append(mean_val)
        if lr_part:
            aggregated_by_r_and_lr[(r_value, model_type, lr_part)].append(mean_val)


# 2. Plot des moyennes par r (tous lr confondus)
plot_data_r  = defaultdict(list)
all_r_values = sorted({r for (r, _) in aggregated_by_r})

for model_type in ["eLM", "iLM"]:
    for r_val in all_r_values:
        vals = aggregated_by_r.get((r_val, model_type), [])
        if vals:
            plot_data_r[model_type].append((r_val, sum(vals)/len(vals)))

plt.figure(figsize=(8, 5))
for model_type, values in plot_data_r.items():
    values = sorted(values)
    x, y   = zip(*values)
    ls     = '-' if model_type == "eLM" else '--'
    plt.plot(x, y, marker='o', linestyle=ls, label=model_type)

# n’afficher la légende que si on a des labels
handles, labels = plt.gca().get_legend_handles_labels()
if labels:
    plt.legend()

plt.xlabel("Taille relative de l'environnement inversé")
plt.ylabel("Biais moyen $B_H$")
plt.title("Biais moyen selon la taille relative (tous lr confondus)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, "moyennes_par_r.png"))
plt.close()


# 3. Plot des moyennes par r ET lr
plot_data_r_lr = defaultdict(list)
for (r_val, model_type, lr_val), vals in aggregated_by_r_and_lr.items():
    avg   = sum(vals)/len(vals)
    label = f"{model_type}, {lr_val}"
    plot_data_r_lr[label].append((r_val, avg))

plt.figure(figsize=(10, 6))
for label, values in plot_data_r_lr.items():
    values = sorted(values)
    x, y   = zip(*values)
    ls     = '-' if "eLM" in label else '--'
    plt.plot(x, y, marker='o', linestyle=ls, label=label)

handles, labels = plt.gca().get_legend_handles_labels()
if labels:
    plt.legend()

plt.xlabel("Taille relative de l'environnement inversé")
plt.ylabel("Biais moyen $B_H$")
plt.title("Biais moyen selon la taille relative (par lr)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, "moyennes_par_r_avec_lr.png"))
plt.close()
