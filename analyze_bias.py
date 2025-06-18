import os, re
import pandas as pd

# dossier où sont vos CSV
bias_dir = "bias_outputs_2500"

# pattern pour extraire seed, p, lr, steps et method depuis le nom de fichier
pattern = re.compile(
    r"split(?P<seed>\d+)_p\d+_lr(?P<lr>[\d.eE+-]+)_steps(?P<steps>\d+)_(?P<method>eLM|iLM)\.csv"
)

records = []
for fname in os.listdir(bias_dir):
    m = pattern.match(fname)
    if not m:
        continue
    info = m.groupdict()
    lr      = float(info["lr"])
    steps   = int(info["steps"])
    method  = info["method"]
    seed    = int(info["seed"])
    # lecture du CSV
    df      = pd.read_csv(os.path.join(bias_dir, fname))
    mean_b  = df["bias"].mean()
    records.append({
        "method": method,
        "lr":      lr,
        "steps":   steps,
        "seed":    seed,
        "mean_bias": mean_b
    })

# on construit le DataFrame complet
df = pd.DataFrame(records)

# 1) on sauve le “long format”
df.to_csv("bias_mean_by_seed_long.csv", index=False)

# 2) on agrège sur seed pour ne garder que la moyenne par (method,lr,steps)
summary = (
    df
    .groupby(["method","lr","steps"], as_index=False)
    ["mean_bias"]
    .mean()
    .rename(columns={"mean_bias":"bias_mean"})
)

# on sauve le résumé
summary.to_csv("bias_mean_by_step_lr.csv", index=False)

# 3) pour une lecture plus rapide, on peut pivoter
pivot = summary.pivot_table(
    index="steps",
    columns=["method","lr"],
    values="bias_mean"
)

# on peut aussi l’exporter
pivot.to_csv("bias_pivot_by_step_lr.csv")

print("Résumé des biais moyens par step / méthode / lr :")
print(pivot)
