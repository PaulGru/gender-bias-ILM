#!/usr/bin/env python3
# 05_plot_from_runs.py (updated)
# - Courbes d'entraînement (moyenne ± std) par mode, échantillonnées tous les N steps (--downsample)
# - Pertes par environnement + "gap" |max - min| pour les modes à environnements (ilm, game, game_phi_fixed)
# - Bar chart BH par paire: par défaut MOYENNE sur TOUTES les proportions (pXXX). Option: --pairs_ptag p050
#
# Exemples:
#   python 05_plot_from_runs.py --runs_root ./runs --steps 2500 --outdir ./figs
#   python 05_plot_from_runs.py --runs_root ./runs --steps 2500 --outdir ./figs --pairs_ptag p050 --downsample 20

import argparse, os, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RUN_RE = re.compile(r"^(erm|ilm|game|game_phi_fixed)_wt2_(p\d{3})_lr([0-9.eE-]+)_s(\d+)_t(\d+)$")
NAME_MAP = {"ilm": "iLM", "erm": "eLM", "game":"IRM-Games", "game_phi_fixed":"IRM-G φ fixed"}
COLOR_MAP = {"ilm":"#ff7f0e", "erm":"#1f77b4", "game":"#2ca02c", "game_phi_fixed":"#9467bd"}

def parse_run_dir(name):
    m = RUN_RE.match(name)
    if not m: return None
    mode, ptag, lr, seed, steps = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
    return {"mode": mode, "ptag": ptag, "lr": lr, "seed": seed, "steps": steps}

def load_total_loss(run_dir):
    from pathlib import Path
    import pandas as pd
    p = Path(run_dir) / "train_total_loss.csv"
    if p.exists():
        df = pd.read_csv(p)
        # attendu: colonnes 'step','loss'
        if {"step","loss"}.issubset(df.columns):
            return df[["step","loss"]]
        return df

    # Fallback pour game_phi_fixed : reconstituer la courbe depuis train_env_losses.csv
    p2 = Path(run_dir) / "train_env_losses.csv"
    if p2.exists():
        df = pd.read_csv(p2)
        if "step" not in df.columns:
            return None
        # toutes colonnes sauf 'step' = environnements (ex: A,B ou loss::A, loss::B)
        env_cols = [c for c in df.columns if c != "step"]
        # conversion numérique prudente
        for c in env_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["loss"] = df[env_cols].mean(axis=1)
        return df[["step","loss"]]

    return None


def load_env_losses(run_dir):
    p = Path(run_dir) / "train_env_losses.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    # attendu: colonnes style 'step','A','B',... ou 'step','loss::A', etc.
    return df

def collect_runs(root, steps_target=None):
    rows = []
    root = Path(root)
    for d in sorted(root.iterdir()):
        if not d.is_dir(): continue
        info = parse_run_dir(d.name)
        if not info: continue
        info["path"] = str(d)
        info["total"] = load_total_loss(d)
        info["envs"]  = load_env_losses(d)
        # bh pairs csv (optionnel, pattern usuel)
        bhcsv = d / f"bh_{info['mode']}_{info['ptag']}_lr{info['lr']}_s{info['seed']}_t{info['steps']}.csv"
        info["bhcsv"] = str(bhcsv) if bhcsv.exists() else None
        rows.append(info)
    return rows

def downsample_df(df, every=10):
    if df is None or df.empty: return df
    return df.iloc[::max(1,int(every)), :].reset_index(drop=True)

def align_and_aggregate_total(runs, mode):
    mats=[]
    for r in runs:
        if r["mode"] != mode or r["total"] is None: continue
        df = r["total"].copy()
        if "step" not in df.columns or "loss" not in df.columns:
            continue
        mats.append(df.set_index("step")["loss"].rename(f"{r['seed']}"))
    if not mats:
        return None
    big = pd.concat(mats, axis=1).sort_index()
    mean = big.mean(axis=1)
    std  = big.std(axis=1).fillna(0.0)
    return pd.DataFrame({"step": mean.index, "mean": mean.values, "std": std.values})

def aggregate_envs(runs, mode):
    """Agrège les pertes par environnement pour un mode donné (ilm/game/game_phi_fixed)"""
    mats=[]; env_names=set()
    for r in runs:
        if r["mode"] != mode or r["envs"] is None: continue
        df = r["envs"].copy()
        if "step" not in df.columns: continue
        cols = [c for c in df.columns if c != "step"]
        ren = {c: (c if "::" in c else f"loss::{c}") for c in cols}
        df = df.rename(columns=ren)
        cols = [c for c in df.columns if c.startswith("loss::")]
        for c in cols: env_names.add(c.split("::",1)[1])
        mats.append(df[["step"]+cols].set_index("step"))
    if not mats:
        return None
    big = pd.concat(mats, axis=0).sort_index()
    mean_df = big.groupby(big.index).mean()
    std_df  = big.groupby(big.index).std().fillna(0.0)
    if len(env_names) >= 2:
        vals = mean_df.values
        gap = np.max(vals, axis=1) - np.min(vals, axis=1)
        gap = pd.Series(gap, index=mean_df.index, name="gap")
    else:
        gap = pd.Series([0.0]*len(mean_df), index=mean_df.index, name="gap")
    return mean_df, std_df, gap

def plot_total_loss_by_mode(runs, outdir, steps_target, downsample):
    present = sorted({r["mode"] for r in runs})
    aggs = {m: align_and_aggregate_total(runs, m) for m in present}
    present = [m for m in present if aggs[m] is not None]
    if not present:
        print("[WARN] Aucun agrégat de total loss disponible."); return
    # figure combinée
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,5))
    for mode in present:
        agg = aggs[mode]
        agg = downsample_df(agg, every=downsample)
        color = COLOR_MAP.get(mode)
        plt.plot(agg["step"], agg["mean"], marker="o", label=f"{NAME_MAP.get(mode,mode)} mean", color=color)
        plt.fill_between(agg["step"], agg["mean"]-agg["std"], agg["mean"]+agg["std"], alpha=0.15, color=color)
    plt.xlabel("step"); plt.ylabel("total loss"); plt.grid(True, alpha=0.2)
    plt.title(f"Average total loss (t={steps_target})")
    plt.legend()
    out = Path(outdir) / f"train_total_loss_ALL_t{steps_target}.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print("[OUT]", out)

    # figures individuelles
    for mode in present:
        agg = aggs[mode]
        agg = downsample_df(agg, every=downsample)
        color = COLOR_MAP.get(mode)
        plt.figure(figsize=(8,5))
        plt.plot(agg["step"], agg["mean"], marker="o", label=f"{NAME_MAP.get(mode,mode)} mean", color=color)
        plt.fill_between(agg["step"], agg["mean"]-agg["std"], agg["mean"]+agg["std"], alpha=0.15, color=color)
        avg_std = float(agg["std"].mean())
        plt.xlabel("step"); plt.ylabel("total loss"); plt.grid(True, alpha=0.2)
        plt.title(f"Average total loss — {NAME_MAP.get(mode,mode)} (t={steps_target})")
        plt.text(0.02, 0.95, f"Avg std across steps: {avg_std:.3f}",
                 transform=plt.gca().transAxes, va="top", ha="left")
        out = Path(outdir) / f"train_total_loss_{mode}_t{steps_target}.png"
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print("[OUT]", out)

def plot_bh_pairs_bar(runs, outdir, steps_target, pairs_ptag=None):
    """
    Bar plot du BH moyen par paire, par mode. Par défaut, moyenne sur toutes les proportions pXXX.
    Si pairs_ptag est donné (ex: p050), on ne garde que cette proportion.
    """
    def gather_one(mode, ptag=None):
        """Concatène tous les CSV de paires pour un mode (et ptag optionnel), ajoute seed, ptag."""
        data=[]
        for r in runs:
            if r["mode"] != mode: continue
            if ptag is not None and r["ptag"] != ptag: continue
            if r["bhcsv"] is None: continue
            try:
                df = pd.read_csv(r["bhcsv"])
                if {"pair","BH_mean"}.issubset(df.columns):
                    if "count" not in df.columns:
                        df["count"] = 1.0
                    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1.0)
                    df = df[["pair","BH_mean","count"]].copy()
                    df["seed"] = r["seed"]
                    df["ptag"] = r["ptag"]
                    data.append(df)
            except Exception:
                continue
        if not data:
            return None
        return pd.concat(data, ignore_index=True)

    present_modes = sorted({r["mode"] for r in runs})
    present_modes = [m for m in present_modes if m in NAME_MAP]

    def aggregate_mode(mode):
        df = gather_one(mode, ptag=pairs_ptag)
        if df is None:
            return None
        # moyenne pondérée par 'count' par paire
        Gm = (df.groupby("pair", as_index=False)
                .apply(lambda g: pd.Series({"BH": float(np.sum(g["BH_mean"]*g["count"]) / np.sum(g["count"]))}))
                .reset_index(drop=True))
        # std non pondérée (approxim.)
        Gs = df.groupby("pair", as_index=False)["BH_mean"].std().rename(columns={"BH_mean":"STD"})
        G = pd.merge(Gm, Gs, on="pair", how="left")
        G["mode"] = mode
        return G

    agg_list=[]
    for m in present_modes:
        a = aggregate_mode(m)
        if a is not None:
            agg_list.append(a)
    if not agg_list:
        print("[WARN] Aucun CSV de paires trouvé."); return

    ALL = pd.concat(agg_list, ignore_index=True)
    pairs = sorted(ALL["pair"].unique().tolist())
    modes = present_modes

    width = 0.8 / max(1,len(modes))
    x = np.arange(len(pairs))
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(max(8, len(pairs)*0.6), 5))
    for i,mode in enumerate(modes):
        sub = ALL[ALL["mode"]==mode].set_index("pair")
        vals = sub.reindex(pairs)["BH"].values
        plt.bar(x + (i - (len(modes)-1)/2)*width, vals, width=width,
                label=NAME_MAP.get(mode,mode), color=COLOR_MAP.get(mode))
    plt.xticks(x, pairs, rotation=45, ha="right")
    plt.ylabel(r"$B_H$ (moyenne par paire)$")
    title_pairs = f"(ALL p)" if pairs_ptag is None else f"({pairs_ptag})"
    plt.title(f"BH par paire {title_pairs} — t={steps_target}")
    plt.legend()
    plt.tight_layout()
    out = Path(outdir) / (f"bh_pairs_bar_{'ALL' if pairs_ptag is None else pairs_ptag}_t{steps_target}.png")
    plt.savefig(out, dpi=150); plt.close()
    print("[OUT]", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--downsample", type=int, default=10)
    ap.add_argument("--pairs_ptag", default=None, help="ex: p050; par défaut: moyenne sur toutes les proportions")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    runs = collect_runs(args.runs_root, steps_target=args.steps)
    if not runs:
        print("[ERR] Aucun run trouvé. Vérifie --runs_root et le pattern des dossiers."); return

    # (1) total loss par mode (downsample)
    plot_total_loss_by_mode(runs, args.outdir, steps_target=args.steps, downsample=args.downsample)

    # (2) bar chart BH par paires (ALL par défaut)
    plot_bh_pairs_bar(runs, args.outdir, steps_target=args.steps, pairs_ptag=args.pairs_ptag)

if __name__ == "__main__":
    main()
