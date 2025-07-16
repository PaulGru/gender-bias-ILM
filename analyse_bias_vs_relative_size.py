import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

means_dir = "output_bias_mean"
plot_output_dir = "plots_par_r"
os.makedirs(plot_output_dir, exist_ok=True)

# Dictionnaires d’agrégation
aggregated_by_r = defaultdict(list)           # clef: (r, model)
aggregated_by_r_and_lr = defaultdict(list)    # clef: (r, model, lr)

for filename in os.listdir(means_dir):
    if not filename.endswith(".csv"):
        continue

    filepath = os.path.join(means_dir, filename)
    # Chargement du CSV et calcul de la moyenne sur **toutes** les valeurs
    df = pd.read_csv(filepath, skiprows=1, header=None)
    mean_val = df.iloc[:,0].astype(float).mean()

    # Extraction de r, lr et model
    name = filename[:-4]  # retire le ".csv"
    parts = name.split("_")
    r_part  = next((p for p in parts if p.startswith("r")),  None)
    lr_part = next((p for p in parts if p.startswith("lr")), None)
    model   = "eLM" if "eLM" in name else "iLM"

    if r_part:
        try:
            r = int(r_part.replace("r", ""))
            aggregated_by_r[(r, model)].append(mean_val)
            if lr_part:
                aggregated_by_r_and_lr[(r, model, lr_part)].append(mean_val)
        except ValueError:
            pass

# --- Tracé 1 : moyennes par r (sans lr) ---
plot_data_r = {"eLM": [], "iLM": []}
all_r = sorted({r for (r, _) in aggregated_by_r})

for model in ["eLM", "iLM"]:
    for r in all_r:
        vals = aggregated_by_r.get((r, model), [])
        if vals:
            plot_data_r[model].append((r, sum(vals)/len(vals)))

plt.figure(figsize=(8,5))
for model, xy in plot_data_r.items():
    xy = sorted(xy)
    x, y = zip(*xy)
    ls = "-" if model=="eLM" else "--"
    plt.plot(x, y, marker="o", linestyle=ls, label=model)

plt.xlabel("Taille relative de l'environnement inversé")
plt.ylabel("Biais moyen $B_H$")
plt.title("Biais moyen selon la taille relative")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, "moyennes_par_r.png"))
plt.close()

# --- Tracé 2 : moyennes par r ET lr ---
plot_data_r_lr = defaultdict(list)
for (r, model, lr), vals in aggregated_by_r_and_lr.items():
    avg = sum(vals)/len(vals)
    label = f"{model}, {lr}"
    plot_data_r_lr[label].append((r, avg))

plt.figure(figsize=(10,6))
for label, xy in plot_data_r_lr.items():
    xy = sorted(xy)
    x, y = zip(*xy)
    ls = "-" if label.startswith("eLM") else "--"
    plt.plot(x, y, marker="o", linestyle=ls, label=label)

plt.xlabel("Taille relative de l'environnement inversé")
plt.ylabel("Biais moyen $B_H$")
plt.title("Biais moyen selon la taille relative et le learning rate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, "moyennes_par_r_avec_lr.png"))
plt.close()
