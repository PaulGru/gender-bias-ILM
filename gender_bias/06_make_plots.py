#!/usr/bin/env python3
# 06_make_plots.py
# Génère 3 graphes “classiques” + 1 figure “Relative sizes (%)” (iLM vs ERM)
#
# Figures produites dans --outdir :
# 1) bh_vs_steps_by_mode.png (+ bh_vs_steps_by_mode.csv)
# 2) bh_vs_steps_by_env_ILM.png (+ ilm_steps_by_rel_ba_pct.csv)
# 3) bh_vs_steps_by_env_ERM.png (+ erm_steps_by_rel_ba_pct.csv)
# 4) fig_rel_sizes.png (Average bias vs Relative sizes, barres = std entre seeds)
#
# Exemples :
#   python 06_make_plots.py --csv results_summary.csv --outdir ./figs
#   python 06_make_plots.py --csv results_summary.csv --outdir ./figs --rel_sizes_step_mode exact:2500

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- utils charge/clean --------------------

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalisation des colonnes
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.lower().str.strip()
    for col in ["rel_ba_pct", "p", "lr", "steps", "seed", "bh_macro"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # garder iLM/ERM uniquement, lignes BH valides
    df = df[df["mode"].isin(["ilm","erm"])].copy()
    df = df[~df["bh_macro"].isna()].copy()
    return df

# -------------------- Figure 1 : BH vs steps (moyenne sur seeds), ILM vs ERM --------------------

def plot_bh_vs_steps_by_mode(df: pd.DataFrame, outdir: str):
    """
    - moyenne sur seeds à l'intérieur de chaque (mode, steps, rel_ba_pct, lr)
    - puis moyenne ET écart-type à travers les (rel_ba_pct, lr) -> un point par (mode, steps)
    """
    seed_mean = (
        df.groupby(["mode","steps","rel_ba_pct","lr"], dropna=False)["bh_macro"]
          .mean()
          .reset_index(name="bh_seed_mean")
    )
    step_mean = (
        seed_mean.groupby(["mode","steps"], dropna=False)["bh_seed_mean"]
        .mean()
        .reset_index(name="bh_mean_over_seeds")
    )
    step_std = (
        seed_mean.groupby(["mode","steps"], dropna=False)["bh_seed_mean"]
        .std()
        .reset_index(name="bh_std_over_seeds")
    )
    step_count = (
        seed_mean.groupby(["mode","steps"], dropna=False)["bh_seed_mean"]
        .size()
        .reset_index(name="n_groups")
    )
    step_agg = (
        step_mean.merge(step_std, on=["mode","steps"], how="left")
                 .merge(step_count, on=["mode","steps"], how="left")
                 .sort_values(["mode","steps"])
    )

    plt.figure(figsize=(9,5))
    name_map = {"ilm": "iLM", "erm": "eLM"}
    for mode in ["ilm","erm"]:
        sub = step_agg[step_agg["mode"]==mode].sort_values("steps")
        if not sub.empty:
            plt.errorbar(sub["steps"],
                         sub["bh_mean_over_seeds"],
                         yerr=sub["bh_std_over_seeds"],
                         fmt='-o', capsize=3, label=f"{name_map[mode]}")
    plt.xlabel("Training steps")
    plt.ylabel("Average bias (avg over rel. sizes & LRs)")
    plt.title("Average bias vs training steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figpath = os.path.join(outdir, "bh_vs_steps_by_mode.png")
    plt.savefig(figpath, dpi=180)
    plt.close()

    csvpath = os.path.join(outdir, "bh_vs_steps_by_mode.csv")
    step_agg.to_csv(csvpath, index=False)
    return figpath, csvpath


def plot_bh_vs_steps_by_mode_no_lr(df: pd.DataFrame, outdir: str):
    """
    Comme plot_bh_vs_steps_by_mode, mais on ne moyenne PAS sur le learning rate.
    Étapes:
      1) moyenne sur seeds pour (mode, steps, rel_ba_pct, lr)
      2) agrégation sur rel_ba_pct -> un point par (mode, steps, lr)
    """
    # 1) moyenne intra-(mode, steps, rel_ba_pct, lr) sur les seeds
    seed_mean = (
        df.groupby(["mode","steps","rel_ba_pct","lr"], dropna=False)["bh_macro"]
          .mean()
          .reset_index(name="bh_seed_mean")
    )

    # 2) agrégation sur les environnements (rel_ba_pct), en conservant lr
    step_mean = (
        seed_mean.groupby(["mode","steps","lr"], dropna=False)["bh_seed_mean"]
        .mean()
        .reset_index(name="bh_mean_over_seeds")
    )
    step_std = (
        seed_mean.groupby(["mode","steps","lr"], dropna=False)["bh_seed_mean"]
        .std()
        .reset_index(name="bh_std_over_seeds")
    )
    step_count = (
        seed_mean.groupby(["mode","steps","lr"], dropna=False)["bh_seed_mean"]
        .size()
        .reset_index(name="n_envs")
    )

    step_agg = (
        step_mean.merge(step_std, on=["mode","steps","lr"], how="left")
                 .merge(step_count, on=["mode","steps","lr"], how="left")
                 .sort_values(["mode","lr","steps"])
    )
    # std NaN -> 0 quand un seul env
    step_agg["bh_std_over_seeds"] = step_agg["bh_std_over_seeds"].fillna(0.0)

    # --- Plot : une courbe par (mode, lr)
    plt.figure(figsize=(9,5))
    name_map = {"ilm":"iLM", "erm":"eLM"}
    for mode in ["ilm","erm"]:
        subm = step_agg[step_agg["mode"]==mode]
        for lr in sorted(subm["lr"].dropna().unique()):
            sub = subm[subm["lr"]==lr].sort_values("steps")
            if sub.empty: 
                continue
            plt.errorbar(
                sub["steps"],
                sub["bh_mean_over_seeds"],
                yerr=sub["bh_std_over_seeds"],
                fmt='-o', capsize=3,
                label=f"{name_map.get(mode, mode.upper())} (lr={lr:g})"
            )

    plt.xlabel("Training steps")
    plt.ylabel("Average bias (avg over relative sizes only)")
    plt.title("Average bias vs training steps — no LR averaging")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    figpath = os.path.join(outdir, "bh_vs_steps_by_mode_noLR.png")
    plt.savefig(figpath, dpi=180)
    plt.close()

    csvpath = os.path.join(outdir, "bh_vs_steps_by_mode_noLR.csv")
    step_agg.to_csv(csvpath, index=False)
    return figpath, csvpath


# -------------------- Figures 2 & 3 : BH vs steps par environnement --------------------

def plot_bh_vs_steps_by_env(df: pd.DataFrame, outdir: str, mode_filter: str):
    """
    - moyenne sur seeds (et sur lrs) pour chaque (mode, rel_ba_pct, steps)
    - courbes : x = steps, y = mean BH, 1 courbe par relative size (B/A %)
    """
    env_seed_mean = (
        df.groupby(["mode","rel_ba_pct","steps"], dropna=False)["bh_macro"]
          .mean()
          .reset_index(name="bh_mean_over_seeds")
    )

    sub = env_seed_mean[env_seed_mean["mode"]==mode_filter].copy()
    if sub.empty:
        return None, None

    plt.figure(figsize=(10,6))
    for rel in sorted(sub["rel_ba_pct"].dropna().unique()):
        chunk = sub[sub["rel_ba_pct"]==rel].sort_values("steps")
        if not chunk.empty:
            plt.plot(chunk["steps"], chunk["bh_mean_over_seeds"], marker='o', label=f"B is {int(rel)}% of A")
    plt.xlabel("Training steps")
    plt.ylabel("Average bias (avg over seeds & LRs)")
    title_name = "iLM" if mode_filter.lower()=="ilm" else "eLM"
    plt.title(f"{title_name} — Average bias vs steps by relative size (Env-B / Env-A)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    figpath = os.path.join(outdir, f"bh_vs_steps_by_env_{mode_filter.upper()}.png")
    plt.savefig(figpath, dpi=180)
    plt.close()

    pivot = sub.pivot_table(index="steps", columns="rel_ba_pct", values="bh_mean_over_seeds")
    csvpath = os.path.join(outdir, f"{mode_filter.lower()}_steps_by_rel_ba_pct.csv")
    pivot.to_csv(csvpath)
    return figpath, csvpath

# -------------------- Figure 4 : Average bias vs Relative sizes (%) (iLM/ERM) --------------------

def select_steps(df: pd.DataFrame, step_mode: str) -> pd.DataFrame:
    """
    step_mode:
      - 'max' : dernier step par (mode, rel_ba_pct, seed, lr)
      - 'mean': moyenne sur tous les steps à l’intérieur de (mode, rel_ba_pct, seed, lr)
      - 'exact:<N>': uniquement les lignes au step N (ex: exact:2500)
    """
    sm = step_mode.strip().lower()
    if sm == "max":
        idx = (df.sort_values("steps")
                 .groupby(["mode","rel_ba_pct","seed","lr"], as_index=False)
                 .tail(1).index)
        return df.loc[idx].copy()
    elif sm == "mean":
        return (df.groupby(["mode","rel_ba_pct","seed","lr"], as_index=False)["bh_macro"]
                  .mean().rename(columns={"bh_macro":"bh_macro"}))
    elif sm.startswith("exact:"):
        try:
            target = int(sm.split(":")[1])
        except Exception:
            raise ValueError("Format attendu pour exact:<N>, ex: --rel_sizes_step_mode exact:2500")
        sub = df[df["steps"] == target].copy()
        if sub.empty:
            raise ValueError(f"Aucune ligne avec steps == {target}.")
        return sub
    else:
        raise ValueError("step_mode doit être 'max', 'mean' ou 'exact:<N>'")

def aggregate_seed_stats(df_sel: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque (mode, rel_ba_pct) :
      - on moyenne d’abord sur les LRs à l’intérieur de chaque seed,
      - puis on calcule moyenne et std ENTRE seeds.
    (Compat pandas anciens : pas de named-agg moderne requis.)
    """
    per_seed = (df_sel.groupby(["mode","rel_ba_pct","seed"], as_index=False)["bh_macro"]
                   .mean().rename(columns={"bh_macro":"bh_per_seed"}))

    g = per_seed.groupby(["mode","rel_ba_pct"])
    tmp = g["bh_per_seed"].agg(["mean","std","count"]).reset_index()
    tmp = tmp.rename(columns={"count":"n"})
    tmp["std"] = tmp["std"].fillna(0.0)  # si une seule seed
    agg = tmp.sort_values(["mode","rel_ba_pct"]).reset_index(drop=True)
    return agg  # colonnes: mode, rel_ba_pct, mean, std, n

def plot_rel_sizes(df: pd.DataFrame, outdir: str, step_mode: str):
    """
    Figure “Average bias vs Relative sizes (%)”
    - 2 courbes : ERM (plein) et iLM (pointillé)
    - barres = écart-type ENTRE seeds
    - selection de step via step_mode
    """
    df_sel = select_steps(df, step_mode)
    agg = aggregate_seed_stats(df_sel)

    plt.figure(figsize=(7,4.2))
    any_line = False
    for mode, style, label in [("erm","-","eLM"), ("ilm","--","iLM")]:
        sub = agg[agg["mode"]==mode]
        if sub.empty:
            continue
        any_line = True
        xs = sub["rel_ba_pct"].values
        ys = sub["mean"].values
        es = sub["std"].values
        plt.errorbar(xs, ys, yerr=es, fmt='o', linestyle=style, capsize=3, label=label)

    plt.xlabel("Relative sizes (Env-B / Env-A in %)")
    plt.ylabel("Average bias")
    plt.title("Average bias vs Relative sizes")
    plt.grid(True, alpha=0.3)
    if any_line:
        plt.legend()
    plt.tight_layout()

    out_path = os.path.join(outdir, "fig_rel_sizes.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Chemin vers results_summary.csv")
    ap.add_argument("--outdir", default=".", help="Répertoire de sortie (figures/CSVs)")
    ap.add_argument("--rel_sizes_step_mode", default="max",
                    help="Sélection des steps pour la figure Relative sizes: 'max' | 'mean' | 'exact:<N>'")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_and_clean(args.csv)

    # (1) ILM vs ERM par steps
    f1, c1 = plot_bh_vs_steps_by_mode(df, args.outdir)
    print("[OUT]", f1); print("[OUT]", c1)

    # (1-bis) Variante: même plot mais SANS moyenne sur le learning rate
    f1b, c1b = plot_bh_vs_steps_by_mode_no_lr(df, args.outdir)
    print("[OUT]", f1b); print("[OUT]", c1b)

    # (2) ILM par environnements
    f2, c2 = plot_bh_vs_steps_by_env(df, args.outdir, mode_filter="ilm")
    if f2: print("[OUT]", f2); print("[OUT]", c2)

    # (3) ERM par environnements
    f3, c3 = plot_bh_vs_steps_by_env(df, args.outdir, mode_filter="erm")
    if f3: print("[OUT]", f3); print("[OUT]", c3)

    # (4) Relative sizes (barres = std entre seeds)
    f4 = plot_rel_sizes(df, args.outdir, step_mode=args.rel_sizes_step_mode)
    print("[OUT]", f4)

if __name__ == "__main__":
    main()
