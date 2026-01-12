#!/usr/bin/env python3
# 04_make_plots.py (updated)
# Génère des figures à partir de results_summary.csv
# - Figure 1: BH vs steps (moyenne sur seeds) pour les modes présents
# - Figure 1b: BH vs steps (moyenne sur seeds), agrégé sur LR (pas de séparations LR)
# - Figure 2+: BH vs steps par environnements pour chaque mode qui utilise des envs (ilm, game, game_phi_fixed)
# - Figure 4: Average bias vs Relative sizes (%) avec sélection du step (max/mean/exact:N)
#
# Exemples :
#   python 04_make_plots.py --csv results_summary.csv --outdir ./figs
#   python 04_make_plots.py --csv results_summary.csv --outdir ./figs --rel_sizes_step_mode exact:2500 --modes ilm,erm,game_phi_fixed

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NAME_MAP = {
    "erm": "eLM",
    "ilm": "iLM",
    "game": "IRM-Games",
    "game_phi_fixed": "IRM-G φ fixed",
}
COLOR_MAP = {
    "erm": "#1f77b4",
    "ilm": "#ff7f0e",
    "game": "#2ca02c",
    "game_phi_fixed": "#9467bd",
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="results_summary.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--modes", default=None,
                    help="Liste des modes à garder (ex: ilm,erm,game_phi_fixed). Par défaut: tous ceux présents dans le CSV.")
    ap.add_argument("--rel_sizes_step_mode", default="max",
                    help="Sélection du step pour la Figure Relative sizes: max | mean | exact:<N>")
    return ap.parse_args()

def normalize(df: pd.DataFrame, keep_modes: List[str]) -> pd.DataFrame:
    df = df.copy()
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.lower().str.strip()
    df = df[df["mode"].isin(keep_modes)].copy()
    df = df[np.isfinite(df["bh_macro"])].copy()
    return df

# -------------------- Figure 1 : BH vs steps (moyenne sur seeds) --------------------

def plot_bh_vs_steps_by_mode(df: pd.DataFrame, outdir: str, keep_modes: List[str]) -> Tuple[str, str]:
    """
    - moyenne sur seeds à l'intérieur de chaque (mode, steps, rel_ba_pct, lr)
    - puis moyenne ET écart-type à travers les (rel_ba_pct, lr) -> un point par (mode, steps)
    """
    seed_mean = (
        df.groupby(["mode","steps","rel_ba_pct","lr"], dropna=False)["bh_macro"]
          .mean()
          .rename("bh_seed_mean")
          .reset_index()
    )
    step_mean = (seed_mean.groupby(["mode","steps"], dropna=False)["bh_seed_mean"].mean().rename("bh_mean").reset_index())
    step_std  = (seed_mean.groupby(["mode","steps"], dropna=False)["bh_seed_mean"].std().rename("bh_std").reset_index())
    step_count= (seed_mean.groupby(["mode","steps"], dropna=False)["bh_seed_mean"].count().rename("n").reset_index())
    agg = (step_mean.merge(step_std, on=["mode","steps"], how="left")
                    .merge(step_count, on=["mode","steps"], how="left")
                    .sort_values(["mode","steps"]))

    os.makedirs(outdir, exist_ok=True)
    csvpath = os.path.join(outdir, "bh_vs_steps_by_mode.csv")
    agg.to_csv(csvpath, index=False)

    figpath = os.path.join(outdir, "bh_vs_steps_by_mode.png")
    plt.figure(figsize=(8,5))
    for mode in keep_modes:
        sub = agg[agg["mode"] == mode]
        if sub.empty: continue
        label = NAME_MAP.get(mode, mode)
        color = COLOR_MAP.get(mode, None)
        plt.plot(sub["steps"], sub["bh_mean"], marker="o", label=label, color=color)
        plt.fill_between(sub["steps"], sub["bh_mean"]-sub["bh_std"], sub["bh_mean"]+sub["bh_std"],
                         alpha=0.15, color=color)
    plt.xlabel("steps")
    plt.ylabel(r"$B_H$ (moyenne sur seeds; barres = std sur p & LR)$")
    plt.title(r"$B_H$ vs steps (agrégé sur p & LR)$")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(figpath, dpi=150)
    plt.close()
    return figpath, csvpath

# Variante sans séparation LR (moyenne sur LR d'abord)

def plot_bh_vs_steps_by_mode_no_lr(df: pd.DataFrame, outdir: str, keep_modes: List[str]) -> Tuple[str, str]:
    seed_mean = (
        df.groupby(["mode","steps","rel_ba_pct","seed"], dropna=False)["bh_macro"]
          .mean()
          .rename("bh_seed_mean")
          .reset_index()
    )
    step_mean = (seed_mean.groupby(["mode","steps"], dropna=False)["bh_seed_mean"].mean().rename("bh_mean").reset_index())
    step_std  = (seed_mean.groupby(["mode","steps"], dropna=False)["bh_seed_mean"].std().rename("bh_std").reset_index())
    step_count= (seed_mean.groupby(["mode","steps"], dropna=False)["bh_seed_mean"].count().rename("n").reset_index())
    agg = (step_mean.merge(step_std, on=["mode","steps"], how="left")
                    .merge(step_count, on=["mode","steps"], how="left")
                    .sort_values(["mode","steps"]))

    os.makedirs(outdir, exist_ok=True)
    csvpath = os.path.join(outdir, "bh_vs_steps_by_mode_noLR.csv")
    agg.to_csv(csvpath, index=False)

    figpath = os.path.join(outdir, "bh_vs_steps_by_mode_noLR.png")
    plt.figure(figsize=(8,5))
    for mode in keep_modes:
        sub = agg[agg["mode"] == mode]
        if sub.empty: continue
        label = NAME_MAP.get(mode, mode)
        color = COLOR_MAP.get(mode, None)
        plt.plot(sub["steps"], sub["bh_mean"], marker="o", label=label, color=color)
        plt.fill_between(sub["steps"], sub["bh_mean"]-sub["bh_std"], sub["bh_mean"]+sub["bh_std"],
                         alpha=0.15, color=color)
    plt.xlabel("steps")
    plt.ylabel(r"$B_H$ (moyenne sur seeds; barres = std sur p)$")
    plt.title(r"$B_H$ vs steps (agrégé sur p)$")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(figpath, dpi=150)
    plt.close()
    return figpath, csvpath

# -------------------- Figure 2+: BH vs steps par environnements (par mode) --------------------

def plot_bh_vs_steps_by_env(df: pd.DataFrame, outdir: str, mode_filter: str):
    """Produit une figure par mode (ERM n'a pas d'environnements)"""
    if mode_filter not in df["mode"].unique():
        return None, None
    sub = df[df["mode"] == mode_filter].copy()
    if sub.empty:
        return None, None
    # moyenne sur seeds pour chaque (steps, rel_ba_pct, lr)
    sub["bh_mean_over_seeds"] = (
        sub.groupby(["steps","rel_ba_pct","lr"])["bh_macro"]
           .transform("mean")
    )
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,5))
    for rel in sorted(sub["rel_ba_pct"].unique()):
        cur = sub[sub["rel_ba_pct"]==rel]
        cur = (cur.groupby(["steps"], as_index=False)["bh_mean_over_seeds"].mean()
                   .sort_values("steps"))
        plt.plot(cur["steps"], cur["bh_mean_over_seeds"], marker="o", label=f"p={int(rel)}%")
    figpath = os.path.join(outdir, f"bh_vs_steps_by_env_{mode_filter.upper()}.png")
    plt.xlabel("steps")
    plt.ylabel(r"$B_H$ (moyenne sur seeds)$")
    plt.title(f"{NAME_MAP.get(mode_filter, mode_filter)} — {r'$B_H$ vs steps'} (par environnements)")
    plt.legend(title="Relative size (%)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(figpath, dpi=150)
    plt.close()

    # CSV (pivot steps x rel_ba_pct)
    pivot = sub.pivot_table(index="steps", columns="rel_ba_pct", values="bh_mean_over_seeds")
    csvpath = os.path.join(outdir, f"{mode_filter.lower()}_steps_by_rel_ba_pct.csv")
    pivot.to_csv(csvpath)
    return figpath, csvpath

# -------------------- Figure 4 : Average bias vs Relative sizes (%) --------------------

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
            raise ValueError(f"Aucune ligne au step {target}")
        return sub
    else:
        raise ValueError("step_mode inconnu. Utilise: max | mean | exact:<N>")

def aggregate_rel_sizes(df_sel: pd.DataFrame) -> pd.DataFrame:
    # moyenne sur LR à seed fixée
    tmp = (df_sel.groupby(["mode","rel_ba_pct","seed"], as_index=False)["bh_macro"]
                 .mean()
                 .rename(columns={"bh_macro":"bh_seed_mean"}))

    if tmp.empty:
        return pd.DataFrame(columns=["mode","rel_ba_pct","mean","std","n"])

    # Agrégation finale par (mode, rel_ba_pct)
    agg = (tmp.groupby(["mode","rel_ba_pct"])["bh_seed_mean"]
              .agg(["mean","std","count"])             # <- compatible pandas
              .reset_index()
              .rename(columns={"count":"n"}))

    # pour rester cohérent avec le reste
    agg = agg.sort_values(["mode","rel_ba_pct"]).reset_index(drop=True)
    return agg  # colonnes: mode, rel_ba_pct, mean, std, n


def plot_rel_sizes(df: pd.DataFrame, outdir: str, step_mode: str, keep_modes: List[str]):
    """
    Trace Average bias vs Relative sizes (%) pour tous les modes choisis.
    Barres = std entre seeds. Moyenne sur LR à seed fixée.
    """
    df_sel = select_steps(df, step_mode)
    agg = aggregate_rel_sizes(df_sel)

    os.makedirs(outdir, exist_ok=True)
    figpath = os.path.join(outdir, "fig_rel_sizes.png")
    plt.figure(figsize=(8,5))
    for mode in keep_modes:
        sub = agg[agg["mode"]==mode]
        if sub.empty: continue
        label = NAME_MAP.get(mode, mode)
        color = COLOR_MAP.get(mode, None)
        x = sub["rel_ba_pct"].astype(int).values
        y = sub["mean"].values
        yerr = sub["std"].values
        plt.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", label=label, color=color, capsize=3)

    plt.xlabel("Relative sizes (%)")
    plt.ylabel(r"$B_H$ — moyenne (± std) sur seeds$")
    plt.title("Average bias vs Relative sizes (%)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figpath, dpi=150)
    plt.close()
    return figpath

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    present_modes = sorted(df["mode"].astype(str).str.lower().str.strip().unique().tolist())
    if args.modes:
        keep_modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    else:
        keep_modes = [m for m in present_modes if m in NAME_MAP]
    df = normalize(df, keep_modes=keep_modes)

    # (1) BH vs steps (tous modes retenus)
    f1, c1 = plot_bh_vs_steps_by_mode(df, args.outdir, keep_modes)
    print("[OUT]", f1); print("[OUT]", c1)

    # (1b) version sans séparation LR
    f1b, c1b = plot_bh_vs_steps_by_mode_no_lr(df, args.outdir, keep_modes)
    print("[OUT]", f1b); print("[OUT]", c1b)

    # (2+) par environnements : pour ilm, game, game_phi_fixed s’ils sont présents
    for mode in keep_modes:
        fp, cp = plot_bh_vs_steps_by_env(df, args.outdir, mode_filter=mode)
        if fp: print("[OUT]", fp); print("[OUT]", cp)

    # (4) Relative sizes (barres = std entre seeds)
    f4 = plot_rel_sizes(df, args.outdir, step_mode=args.rel_sizes_step_mode, keep_modes=keep_modes)
    print("[OUT]", f4)

if __name__ == "__main__":
    main()
