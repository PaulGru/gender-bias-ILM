import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

means_dir      = "output_bias_mean"
plot_output_dir = "plots_par_step"
os.makedirs(plot_output_dir, exist_ok=True)

target_r_values = ["r30", "r70", None]

for target_r in target_r_values:
    aggregated = defaultdict(list)

    for filename in os.listdir(means_dir):
        # 1) on ne garde que les CSV
        if not filename.lower().endswith(".csv"):
            continue
        # 2) filtrage sur rX
        if target_r is not None and target_r not in filename:
            continue
        if target_r is None and "r" not in filename:
            continue

        filepath = os.path.join(means_dir, filename)
        # 3) lecture avec pandas
        df = pd.read_csv(filepath)
        # 4) récupération de la moyenne
        mean_val = df['bias'].iloc[0] if 'bias' in df.columns else df.iloc[0,0]

        # Extraction du step et du type de modèle
        parts     = filename.split("_")
        step_part = next((p for p in parts if p.startswith("steps")), None)
        model_type= "eLM" if "eLM" in filename else "iLM"

        if step_part:
            step = int(step_part.replace("steps", ""))
            aggregated[(step, model_type)].append(mean_val)

    # Prépare le data pour le plot
    plot_data = defaultdict(list)
    all_steps = sorted({step for (step, _) in aggregated})

    for model_type in ["eLM", "iLM"]:
        for step in all_steps:
            vals = aggregated.get((step, model_type), [])
            if vals:
                plot_data[model_type].append((step, sum(vals)/len(vals)))

    # Tracé
    plt.figure(figsize=(8, 5))
    for mtype, vals in plot_data.items():
        x, y = zip(*sorted(vals))
        ls = '-' if mtype == "eLM" else '--'
        plt.plot(x, y, marker='o', linestyle=ls, label=mtype)

    plt.xlabel("Nombre de steps d'entraînement")
    plt.ylabel("Biais moyen $B_H$")
    title = "Contrôle du biais de genre"
    title += f" (r = {target_r[1:]})" if target_r else " (tous r confondus)"
    plt.title(title)

    # on ne demande la légende que si on a au moins une courbe
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()

    plt.grid(True)
    plt.tight_layout()
    suffix = f"_r{target_r[1:]}" if target_r else "_all_r"
    plt.savefig(os.path.join(plot_output_dir, f"moyennes_par_step{suffix}.png"))
    plt.close()
