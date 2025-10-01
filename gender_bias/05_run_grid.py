#!/usr/bin/env python3
# 05_run_grid.py — grille conforme papier ILM : rel_ba_pct × lrs × steps × seeds
import argparse, os, sys, csv, subprocess, math
from pathlib import Path
import shutil  # <- ajoute ceci

def cleanup_model_dir(model_dir: str):
    import os
    keep_ext = {".csv", ".txt", ".json"}
    keep_names = {
        "config.json", "special_tokens_map.json",
        "tokenizer_config.json", "trainer_state.json",
        "vocab.txt","merges.txt","tokenizer.json"
    }
    for name in os.listdir(model_dir):
        p = os.path.join(model_dir, name)
        if os.path.isdir(p) and name.startswith("checkpoint-"):
            shutil.rmtree(p); continue
        if name in keep_names or os.path.splitext(name)[1] in keep_ext:
            continue
        try:
            os.remove(p)
        except IsADirectoryError:
            shutil.rmtree(p)

def p_from_rel_ba_pct(rel_ba_pct: float) -> float:
    r = rel_ba_pct / 100.0  # B/A
    return 1.0 / (1.0 + r)  # p = A/(A+B)

def ptag_from_p(p: float) -> str:
    return f"p{int(round(p*100)):03d}"

def ensure_data(out_root, rel_ba_pct, data_seed):
    p = p_from_rel_ba_pct(rel_ba_pct)
    ptag = ptag_from_p(p)
    env_dir = Path(out_root) / f"envs_{ptag}"
    erm_dir = Path(out_root) / f"erm_{ptag}"
    if env_dir.exists() and erm_dir.exists():
        print(f"[DATA] reuse {env_dir} / {erm_dir}")
        return p, ptag
    cmd = [
        sys.executable, "01_make_wt2_gender_envs.py",
        "--out_root", out_root,
        "--rel_ba_pct", str(rel_ba_pct),
        "--seed", str(data_seed),
    ]
    print("[DATA]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return p, ptag

def train(model_base, mode, data_root, ptag, steps, seed, lr, bsz=32, gradient_accumulation_steps=1, out_root="runs", extra_args=None):
    train_dir = Path(data_root) / ("envs_"+ptag if mode=="ilm" else "erm_"+ptag)
    valid_file = Path(data_root) / "valid.txt"
    out_dir = Path(out_root) / f"{mode}_wt2_{ptag}_lr{lr}_s{seed}_t{steps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "run_invariant_mlm.py",
        "--model_name_or_path", model_base,
        "--train_file", str(train_dir),
        "--validation_file", str(valid_file),
        "--do_train", "--do_eval",
        "--output_dir", str(out_dir),
        "--per_device_train_batch_size", str(bsz),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(lr),
        "--nb_steps", str(steps),
        "--logging_steps", "50",
        "--save_steps", "0",
        "--seed", str(seed)
    ]
    if mode == "ilm":
        cmd += ["--mode", "ilm"]
    if extra_args:
        cmd += extra_args
    print("[TRAIN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)
    return str(out_dir)

def eval_bh(model_dir, test_file, outfile_pairs, dump_used=None):
    cmd = [sys.executable, "04_eval_bh.py",
           "--model_dir", model_dir,
           "--test_file", test_file,
           "--max_samples", "100000",
           "--outfile", outfile_pairs]
    if dump_used:
        cmd += ["--dump_used", dump_used]
    print("[EVAL]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def read_macro_from_pairs(pairs_csv):
    import csv
    vals = []
    with open(pairs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vals.append(float(row["BH_mean"]))
            except Exception:
                pass
    return (sum(vals)/len(vals)) if vals else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel_ba_pcts", default="10,25,30,50,70,75,90,100",
                    help="Liste de relative sizes B/A en PERCENT (papier).")
    ap.add_argument("--lrs", default="1e-5,5e-5")
    ap.add_argument("--steps_list", default="10,50,100,200,1000,2500")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--data_seed", type=int, default=0, help="seed pour le split des env (fixé)")
    ap.add_argument("--data_root", default="./data/wt2_gender")
    ap.add_argument("--model_base", default="distilbert-base-uncased")
    ap.add_argument("--runs_root", default="./runs")
    ap.add_argument("--results_csv", default="./results_summary.csv")
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Nombre de mises à jour de gradient à accumuler avant de faire un pas d'optimisation.")
    ap.add_argument("--extra_train_args", default="", help="ex: \"--fp16 --max_seq_length 128\"")
    ap.add_argument("--eval_dump_used", action="store_true")
    ap.add_argument("--cleanup", action="store_true",
                help="Delete heavy model files in each run dir after BH eval.")
    args = ap.parse_args()

    rel_ba_pcts = [float(x) for x in args.rel_ba_pcts.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")]
    steps_list = [int(x) for x in args.steps_list.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    extra_args = args.extra_train_args.split() if args.extra_train_args.strip() else []

    results_path = Path(args.results_csv)
    write_header = not results_path.exists()
    with open(results_path, "a", newline="", encoding="utf-8") as outf:
        wr = csv.writer(outf)
        if write_header:
            wr.writerow(["mode","rel_ba_pct","p","lr","steps","seed","pairs_csv","bh_macro","model_dir"])
        test_file = str(Path(args.data_root) / "test.txt")
        for rel_ba_pct in rel_ba_pcts:
            p, ptag = ensure_data(args.data_root, rel_ba_pct, data_seed=args.data_seed)
            for lr in lrs:
                for steps in steps_list:
                    for seed in seeds:
                        for mode in ["ilm","erm"]:
                            model_dir = train(args.model_base, mode, args.data_root, ptag, steps, seed, lr,
                                              bsz=args.bsz, out_root=args.runs_root, extra_args=extra_args)
                            pairs_csv = str(Path(model_dir) / f"bh_{mode}_{ptag}_lr{lr}_s{seed}_t{steps}.csv")
                            dump_csv  = str(Path(model_dir) / f"bh_used_{mode}_{ptag}_lr{lr}_s{seed}_t{steps}.csv") if args.eval_dump_used else None
                            eval_bh(model_dir, test_file, pairs_csv, dump_used=dump_csv)
                            cleanup_model_dir(model_dir)
                            bh_macro = read_macro_from_pairs(pairs_csv)
                            wr.writerow([mode, rel_ba_pct, p, lr, steps, seed, pairs_csv, bh_macro, model_dir])
                            outf.flush()
                            print(f"[RESULT] mode={mode} rel_ba={rel_ba_pct}% p={p:.3f} lr={lr} steps={steps} seed={seed} macro={bh_macro:.4f}")

if __name__ == "__main__":
    main()
