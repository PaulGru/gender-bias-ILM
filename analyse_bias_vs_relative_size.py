import os
import matplotlib.pyplot as plt
from collections import defaultdict

means_dir = "bias_means"
# Dossier de sauvegarde pour les plots par r
plot_output_dir = "plots_par_r"
os.makedirs(plot_output_dir, exist_ok=True)


# 1. Agréger les données depuis les fichiers *_mean.txt
aggregated_by_r = defaultdict(list)           # (r, model)
aggregated_by_r_and_lr = defaultdict(list)    # (r, model, lr)

for filename in os.listdir(means_dir):
    if filename.endswith("_mean.txt"):
        filepath = os.path.join(means_dir, filename)

        # Lire la moyenne
        with open(filepath, "r") as f:
            try:
                mean_val = float(f.read().strip())
            except ValueError:
                continue

        # Extraire r, modèle et lr
        parts = filename.split("_")
        r_part = next((p for p in parts if p.startswith("r")), None)
        lr_part = next((p for p in parts if p.startswith("lr")), None)
        model_type = "eLM" if "eLM" in filename else "iLM"

        if r_part:
            try:
                r_value = int(r_part.replace("r", ""))
                aggregated_by_r[(r_value, model_type)].append(mean_val)
                if lr_part:
                    aggregated_by_r_and_lr[(r_value, model_type, lr_part)].append(mean_val)
            except ValueError:
                continue

# 2. Tracer moyennes par r (sans distinction de lr)
plot_data_r = defaultdict(list)
all_r_values = sorted(set(k[0] for k in aggregated_by_r))

for model_type in ["eLM", "iLM"]:
    for r_val in all_r_values:
        values = aggregated_by_r.get((r_val, model_type), [])
        if values:
            avg = sum(values) / len(values)
            plot_data_r[model_type].append((r_val, avg))

plt.figure(figsize=(8, 5))
for model_type, values in plot_data_r.items():
    values = sorted(values)
    x, y = zip(*values)
    linestyle = '-' if model_type == "eLM" else '--'
    plt.plot(x, y, marker='o', linestyle=linestyle, label=model_type)


plt.xlabel("Taille relative de l'environnement inversé par rapport à l'environnement inchangé")
plt.ylabel("Biais moyen B_{H}")
plt.title("Biais moyen selon la taille relative")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, "moyennes_par_r.png"))
plt.close()


# 3. Tracer moyennes par r ET lr
plot_data_r_lr = defaultdict(list)
all_r_values_lr = sorted(set(k[0] for k in aggregated_by_r_and_lr))

for (r_val, model_type, lr_val), values in aggregated_by_r_and_lr.items():
    avg = sum(values) / len(values)
    label = f"{model_type}, {lr_val}"
    plot_data_r_lr[label].append((r_val, avg))

plt.figure(figsize=(10, 6))
for label, values in plot_data_r_lr.items():
    values = sorted(values)
    x, y = zip(*values)
    linestyle = '-' if "eLM" in label else '--'
    plt.plot(x, y, marker='o', linestyle=linestyle, label=label)


plt.xlabel("Taille relative de l'environnement inversé par rapport à l'environnement inchangé")
plt.ylabel("Biais moyen B_{H}")
plt.title("Biais moyen selon la taille relative")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, "moyennes_par_r_avec_lr.png"))
plt.close()
