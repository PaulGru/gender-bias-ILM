#!/usr/bin/env python3
# 07_plot_from_runs.py
# - Courbes d'entraînement (moyenne ± std) iLM vs ERM, échantillonnées tous les N steps (--downsample)
# - iLM: pertes par environnement + "gap" |A-B|
# - Scatter: BH macro final vs gap final
# - Bar chart BH par paire: par défaut MOYENNE sur TOUTES les proportions (pXXX). Option: --pairs_ptag p050
#
# Exemples:
#   python 07_plot_from_runs.py --runs_root ./runs --steps 2500 --outdir ./figs
#   python 07_plot_from_runs.py --runs_root ./runs --steps 2500 --outdir ./figs --pairs_ptag p050 --downsample 20

import argparse, os, re, shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RUN_RE = re.compile(r"^(erm|ilm)_wt2_(p\d{3})_lr([0-9.eE-]+)_s(\d+)_t(\d+)$")
NAME_MAP = {"ilm": "iLM", "erm": "eLM"}

def parse_run_dir(name):
    m = RUN_RE.match(name)
    if not m: return None
    mode, ptag, lr, seed, steps = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
    return {"mode": mode, "ptag": ptag, "lr": lr, "seed": seed, "steps": steps}

def load_total_loss(run_dir):
    p = Path(run_dir) / "train_total_loss.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    if "step" not in df.columns or "loss" not in df.columns: return None
    return df[["step","loss"]].copy()

def load_env_losses(run_dir):
    p = Path(run_dir) / "train_env_losses.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    if "step" not in df.columns: return None
    cols = [c for c in df.columns if c != "step"]
    if not cols: return None
    return df[["step"] + cols].copy()

def find_bh_csv(run_dir):
    files = list(Path(run_dir).glob("bh_*.csv"))
    return files[0] if files else None

def bh_macro_from_pairs_csv(path):
    try:
        df = pd.read_csv(path)
        if "BH_mean" in df.columns:
            v = pd.to_numeric(df["BH_mean"], errors="coerce").dropna().values
            return float(np.mean(v)) if len(v) > 0 else np.nan
    except Exception:
        pass
    return np.nan

def collect_runs(runs_root, steps_target):
    rows=[]
    for d in Path(runs_root).iterdir():
        if not d.is_dir(): continue
        info = parse_run_dir(d.name)
        if not info: continue
        if info["steps"] != steps_target: continue
        total = load_total_loss(d)
        envs  = load_env_losses(d)
        bhcsv = find_bh_csv(d)
        bhmac = bh_macro_from_pairs_csv(bhcsv) if bhcsv else np.nan
        rows.append({**info, "path": str(d), "total": total, "envs": envs,
                     "bhcsv": str(bhcsv) if bhcsv else None, "bh_macro": bhmac})
    return rows

# ---------- helpers ----------

def downsample_df(df, every):
    """Garde les lignes où step % every == 0 + la dernière ligne."""
    if df is None or df.empty: return df
    last = df["step"].max()
    mask = (df["step"] % every == 0) | (df["step"] == last)
    return df.loc[mask].copy()

def align_and_aggregate_total(runs, mode):
    series=[]
    for r in runs:
        if r["mode"] != mode or r["total"] is None: continue
        s = r["total"].set_index("step")["loss"]
        s.name = r["path"]
        series.append(s)
    if not series: return None
    M = pd.concat(series, axis=1).sort_index()
    return pd.DataFrame({
        "step": M.index,
        "mean": M.mean(axis=1, skipna=True).values,
        "std":  M.std(axis=1, skipna=True).fillna(0.0).values,
        "n":    M.count(axis=1).values
    })

def aggregate_ilm_envs(runs):
    mats=[]; env_names=set()
    for r in runs:
        if r["mode"] != "ilm" or r["envs"] is None: continue
        df = r["envs"].copy()
        cols = [c for c in df.columns if c != "step"]
        if len(cols) == 1:
            df = df.rename(columns={cols[0]: "all"}); cols=["all"]
        for c in cols: env_names.add(c)
        df = df.set_index("step")
        df.columns = [f"{r['path']}::{c}" for c in cols]
        mats.append(df)
    if not mats: return None, None
    big = pd.concat(mats, axis=1).sort_index()
    # per-env mean/std
    env_means={}; env_stds={}
    for env in sorted(env_names):
        cols = [c for c in big.columns if c.endswith(f"::{env}")]
        if not cols: continue
        sub = big[cols]
        env_means[env] = sub.mean(axis=1, skipna=True)
        env_stds[env]  = sub.std(axis=1, skipna=True).fillna(0.0)
    steps = big.index
    mean_df = pd.DataFrame({"step": steps}); std_df = pd.DataFrame({"step": steps})
    for env in sorted(env_means.keys()):
        mean_df[f"mean_{env}"] = env_means[env].values
        std_df[f"std_{env}"]   = env_stds[env].values
    # gap = max(mean_envs) - min(mean_envs) si >=2 envs
    env_cols = [c for c in mean_df.columns if c.startswith("mean_")]
    if len(env_cols) >= 2:
        vals = mean_df[env_cols].values
        mean_df["gap_mean"] = vals.max(axis=1) - vals.min(axis=1)
    return mean_df, std_df

# ---------- plots ----------

def plot_total_loss_by_mode(runs, outdir, steps_target, downsample):
    # Agrège d'abord pour iLM et eLM, afin de partager la même échelle Y et comparer visuellement la variance
    aggs = {m: align_and_aggregate_total(runs, m) for m in ["ilm","erm"]}
    present = [m for m,a in aggs.items() if a is not None and not a.empty]
    if not present: 
        return
    for m in present:
        aggs[m] = downsample_df(aggs[m], downsample)
    # bornes Y communes = [min(mean-std), max(mean+std)] ± 5%
    ymin = min(float((aggs[m]["mean"] - aggs[m]["std"]).min()) for m in present)
    ymax = max(float((aggs[m]["mean"] + aggs[m]["std"]).max()) for m in present)
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ymin -= pad; ymax += pad
    # moyenne des écarts-types (sur les steps) pour annotation
    avg_std = {m: float(np.nanmean(aggs[m]["std"])) for m in present}
    for mode in present:
        agg = aggs[mode]
        plt.figure(figsize=(8,5))
        plt.plot(agg["step"], agg["mean"], marker="o", label=f"{NAME_MAP[mode]} mean")
        plt.fill_between(agg["step"], agg["mean"]-agg["std"], agg["mean"]+agg["std"], alpha=0.2)
        plt.ylim(ymin, ymax)
        plt.xlabel("Training step")
        plt.ylabel("Total loss (avg over seeds & LRs)")
        plt.title(f"Average total loss — {NAME_MAP[mode]} (t={steps_target})")
        # met en évidence la variance par une annotation claire
        plt.text(0.02, 0.95, f"Avg std across steps: {avg_std[mode]:.3f}",
                 transform=plt.gca().transAxes, va="top", ha="left",
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        out = Path(outdir) / f"train_total_loss_{mode}_t{steps_target}.png"
        plt.savefig(out, dpi=180); plt.close()
        print("[OUT]", out)


def plot_bh_pairs_bar(runs, outdir, steps_target, pairs_ptag, pairs_weighted=True, pairs_split_by_p=False):
    """
    Si pairs_ptag == 'ALL' => on agrège sur TOUTES les proportions (pXXX),
    sinon on filtre sur un p précis (ex: 'p050').

    pairs_weighted: si True, moyenne pondérée par 'count' (du CSV BH par paires).
    pairs_split_by_p: si True, on trace des barres séparées par p (couleur = mode, groupe = p).
    """
    def gather_one(mode, ptag=None):
        rows = []
        for r in runs:
            if r["mode"] != mode: 
                continue
            if ptag is not None and r["ptag"] != ptag:
                continue
            if r["bhcsv"] is None:
                continue
            try:
                df = pd.read_csv(r["bhcsv"])
                # attendu: colonnes "pair","count","BH_mean"
                if {"pair","BH_mean"}.issubset(df.columns):
                    # si 'count' absent ou NaN -> 1
                    if "count" not in df.columns:
                        df["count"] = 1
                    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1.0)
                    df = df[["pair","BH_mean","count"]].copy()
                    df["seed"] = r["seed"]
                    df["ptag"] = r["ptag"]
                    rows.append(df)
            except Exception:
                pass
        if not rows:
            return None
        X = pd.concat(rows, ignore_index=True)
        return X

    # --- cas 1: on veut des barres séparées par p ---
    if pairs_split_by_p:
        # on ignore pairs_ptag ici, on affiche tous les p présents
        ilm_all = gather_one("ilm", ptag=None)
        erm_all = gather_one("erm", ptag=None)
        if ilm_all is None and erm_all is None:
            print("[WARN] pas de données BH pour barres par p."); 
            return

        # liste ordonnée de p
        all_ptags = sorted(set(([] if ilm_all is None else ilm_all["ptag"].unique().tolist()) +
                               ([] if erm_all is None else erm_all["ptag"].unique().tolist())))
        # pivot: pair × (mode,p)
        def agg_by_p(df, mode_name):
            if df is None: 
                return pd.DataFrame(columns=["pair","ptag",f"{mode_name}_BH"])
            if pairs_weighted:
                # moyenne pondérée par count: sum(BH_mean*count)/sum(count) au niveau (pair, ptag)
                G = df.groupby(["pair","ptag"], as_index=False).apply(
                    lambda d: pd.Series({
                        f"{mode_name}_BH": np.average(d["BH_mean"], weights=d["count"])
                    })
                ).reset_index(drop=True)
            else:
                G = df.groupby(["pair","ptag"], as_index=False)["BH_mean"].mean().rename(columns={"BH_mean": f"{mode_name}_BH"})
            return G

        I = agg_by_p(ilm_all, "ILM")
        E = agg_by_p(erm_all, "ERM")
        M = pd.merge(I, E, on=["pair","ptag"], how="outer").sort_values(["ptag","pair"])

        # tracé: groupes par ptag, barres ILM/ERM côte à côte
        ptags = M["ptag"].unique().tolist()
        pairs = sorted(M["pair"].unique().tolist())
        x = np.arange(len(pairs))
        width = 0.35
        ncols = max(1, min(3, len(ptags)))
        nrows = int(np.ceil(len(ptags)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(10, len(pairs)*0.45), 4*nrows), squeeze=False, sharey=True)
        for i, p in enumerate(ptags):
            ax = axes[i//ncols][i%ncols]
            sub = M[M["ptag"]==p].set_index("pair")
            ilm_vals = sub.get("ILM_BH", pd.Series(index=pairs, dtype=float)).reindex(pairs)
            erm_vals = sub.get("ERM_BH", pd.Series(index=pairs, dtype=float)).reindex(pairs)
            ax.bar(x - width/2, ilm_vals.values, width=width, label="iLM")
            ax.bar(x + width/2, erm_vals.values, width=width, label="eLM")
            ax.set_title(f"{p}")
            ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.3)
        axes[0][0].set_ylabel("Average bias")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"Average bias by pair — split by p (t={steps_target})", y=1.02)
        fig.tight_layout()
        out = Path(outdir) / f"bh_pairs_bar_split_by_p_t{steps_target}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
        print("[OUT]", out)
        return

    # --- cas 2: une seule barre par pair (agrégée sur p / seeds) ---
    def aggregate_mode(mode):
        if pairs_ptag.upper() == "ALL":
            df = gather_one(mode, ptag=None)
        else:
            df = gather_one(mode, ptag=pairs_ptag)
        if df is None: 
            return None
        if pairs_weighted:
            # moyenne pondérée + un écart-type (non pondéré) pour visualiser la variance entre runs
            def agg_fn(d):
                wmean = np.average(d["BH_mean"], weights=d["count"])
                s     = float(np.std(d["BH_mean"]))
                return pd.Series({f"{mode.upper()}_BH": wmean, f"{mode.upper()}_STD": s})
            G = df.groupby("pair", as_index=False).apply(agg_fn).reset_index(drop=True)
        else:
            Gm = df.groupby("pair", as_index=False)["BH_mean"].mean().rename(columns={"BH_mean": f"{mode.upper()}_BH"})
            Gs = df.groupby("pair", as_index=False)["BH_mean"].std().rename(columns={"BH_mean": f"{mode.upper()}_STD"})
            G  = pd.merge(Gm, Gs, on="pair", how="left")
        return G

    ilm = aggregate_mode("ilm")
    erm = aggregate_mode("erm")
    if ilm is None and erm is None:
        print("[WARN] pas de BH par paires à agréger."); 
        return
    Y = (pd.merge(ilm, erm, on="pair", how="outer") if (ilm is not None and erm is not None)
         else (ilm if ilm is not None else erm))
    Y = Y.sort_values("pair").reset_index(drop=True)
    pairs = Y["pair"].tolist()
    # ajoute le nombre d'exemples en suffixe: "pair (count)"
    pair_counts = {
        "actor/actress":3, "boy/girl":8, "boys/girls":3, "father/mother":1, "he/she":63, "him/her":21,
        "his/her":61, "husband/wife":1, "king/queen":3, "male/female":4, "man/woman":15,
        "prince/princess":5, "son/daughter":3
    }
    labels = [f"{p} ({pair_counts.get(p, '?')})" for p in pairs]
    x = np.arange(len(pairs)); width = 0.38
    plt.figure(figsize=(max(9, len(pairs)*0.5), 5))
    if "ILM_BH" in Y.columns: plt.bar(x - width/2, Y["ILM_BH"], width=width, label="iLM")
    if "ERM_BH" in Y.columns: plt.bar(x + width/2, Y["ERM_BH"], width=width, label="eLM")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Average bias")
    title_ptag = "ALL proportions (weighted)" if (pairs_ptag.upper()=="ALL" and pairs_weighted) else \
                 ("ALL proportions" if pairs_ptag.upper()=="ALL" else pairs_ptag)
    plt.title(f"Average bias by pair — {title_ptag} (t={steps_target})")
    plt.grid(True, axis="y", alpha=0.3); plt.legend(); plt.tight_layout()
    out = Path(outdir) / f"bh_pairs_bar_{pairs_ptag.upper()}_t{steps_target}.png"
    plt.savefig(out, dpi=180); plt.close()
    print("[OUT]", out)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="./runs")
    ap.add_argument("--steps", type=int, default=2500, help="N steps des runs à considérer (suffixe _tN)")
    ap.add_argument("--outdir", default="./figs")
    ap.add_argument("--downsample", type=int, default=10, help="Afficher un point tous les N steps")
    ap.add_argument("--pairs_ptag", default="ALL",
                    help="pTag pour le bar chart (ex: p050). Par défaut 'ALL' = moyenne sur toutes les proportions.")
    ap.add_argument("--pairs_weighted", action="store_true", default=True,
                help="Moyenne pondérée par 'count' (BH eval) pour les barres par paires (par défaut: True).")
    ap.add_argument("--pairs_split_by_p", action="store_true",
                help="Si présent, affiche des barres séparées par p (au lieu d'une moyenne sur tous les p).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    runs = collect_runs(args.runs_root, steps_target=args.steps)
    if not runs:
        print("[ERR] Aucun run trouvé. Vérifie --runs_root et le pattern des dossiers."); return

    # (1) total loss iLM/ERM (downsample)
    plot_total_loss_by_mode(runs, args.outdir, steps_target=args.steps, downsample=args.downsample)
    # (2) bar chart BH par paires (ALL par défaut)
    plot_bh_pairs_bar(runs, args.outdir, steps_target=args.steps, pairs_ptag=args.pairs_ptag)

if __name__ == "__main__":
    main()
