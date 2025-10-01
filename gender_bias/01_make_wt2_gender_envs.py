#!/usr/bin/env python3
# 01_make_wt2_gender_envs.py
# Crée les environnements A (original) et B (swappé) pour WikiText-2, avec un ratio p.
# Sources possibles pour les données :
#   (1) fichiers locaux --local_train/--local_valid/--local_test
#   (2) Parquet HTTP Hugging Face (fallback sans S3)
#   (3) datasets.load_dataset("wikitext","wikitext-2-raw-v1")

import argparse, os, random, re
from pathlib import Path

def build_gender_maps():
    base_pairs = [
        ("he","she"),("him","her"),("his","her"),
        ("man","woman"),("men","women"),("male","female"),
        ("boy","girl"),("boys","girls"),
        ("father","mother"),("fathers","mothers"),
        ("son","daughter"),("sons","daughters"),
        ("brother","sister"),("brothers","sisters"),
        ("uncle","aunt"),("uncles","aunts"),
        ("husband","wife"),("husbands","wives"),
        ("king","queen"),("kings","queens"),
        ("prince","princess"),("princes","princesses"),
        ("actor","actress"),("actors","actresses"),
        ("hero","heroine"),("heroes","heroines"),
        ("gentleman","lady"),("gentlemen","ladies"),
        ("sir","madam"),("sirs","madams"),
        ("spokesman","spokeswoman"),("spokesmen","spokeswomen"),
        ("stepfather","stepmother"),("stepfathers","stepmothers"),
        ("grandson","granddaughter"),("grandsons","granddaughters"),
        ("mr","mrs"),
    ]
    to_f = {a.lower(): b.lower() for a,b in base_pairs}
    to_m = {b.lower(): a.lower() for a,b in base_pairs}
    full = {**to_f, **to_m}
    pattern = r"\b(?:%s)\b" % "|".join(sorted(map(re.escape, full.keys()), key=len, reverse=True))
    return full, re.compile(pattern, flags=re.IGNORECASE)

def apply_case_like(src: str, dst_lower: str) -> str:
    if src.isupper(): return dst_lower.upper()
    if len(src) >= 2 and src[0].isupper() and src[1:].islower(): return dst_lower.capitalize()
    return dst_lower.lower()

def swap_gender(text: str, mapping, pattern) -> str:
    def repl(m):
        w = m.group(0)
        tgt = mapping[w.lower()]
        return apply_case_like(w, tgt)
    return pattern.sub(repl, text)

def read_lines(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for s in f:
            s = s.rstrip("\n")
            if s.strip():
                out.append(s)
    return out

def extract_text_list(ds_split):
    """
    ds_split : datasets.Dataset
    Retourne une liste de strings (colonne 'text' si dispo, sinon concat des colonnes en str).
    Compatible datasets 1.13.x.
    """
    texts = []
    # Itérer ligne par ligne évite d'allouer des listes énormes avec ds_split['text']
    for x in ds_split:
        if isinstance(x, dict):
            if "text" in x and isinstance(x["text"], str):
                s = x["text"]
            else:
                # fallback très permissif
                s = " ".join(str(v) for v in x.values())
        else:
            s = str(x)
        s = s.strip("\n")
        if s.strip():
            texts.append(s)
    return texts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", required=True, help="Dossier de sortie, ex: ./data/wt2_gender")
    p.add_argument("--rel_ba_pct", type=float, default=None,
               help="Relative size B/A en PERCENT (papier ILM): 100 = équilibré (p=0.5), 10 => B/A=0.1.")
    p.add_argument("--rel_size", type=float, default=None,
               help="Ancien ratio A/B (optionnel, compat).")
    p.add_argument("--p", type=float, default=None, help="Ratio d'exemples pour Env-A (original). Env-B reçoit 1-p (swappé). Si --rel_size est fourni, ignoré.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_len", type=int, default=1)
    # Mode LOCAL
    p.add_argument("--local_train", type=str, default=None, help="Chemin vers wiki.train.raw (optionnel)")
    p.add_argument("--local_valid", type=str, default=None, help="Chemin vers wiki.valid.raw (optionnel)")
    p.add_argument("--local_test",  type=str, default=None, help="Chemin vers wiki.test.raw  (optionnel)")
    args = p.parse_args()

    # --- CALCUL de p ---
    if args.rel_ba_pct is not None:
        if args.rel_ba_pct <= 0:
            raise SystemExit("--rel_ba_pct doit être > 0.")
        r = args.rel_ba_pct / 100.0        # r = B/A
        p_value = 1.0 / (1.0 + r)          # p = A / (A+B)
    elif args.rel_size is not None:
        if args.rel_size <= 0:
            raise SystemExit("--rel_size (A/B) doit être > 0.")
        p_value = args.rel_size / (1.0 + args.rel_size)
    elif args.p is not None:
        p_value = args.p
    else:
        p_value = 0.5

    if not (0.0 < p_value < 1.0):
        raise SystemExit(f"p doit être dans (0,1), reçu {p_value}")

    A_over_B = p_value / (1.0 - p_value)
    B_over_A = (1.0 - p_value) / p_value
    print(f"[SPLIT] p={p_value:.4f}  (B/A={B_over_A:.4f} -> {B_over_A*100:.1f}%,  A/B={A_over_B:.4f})")


    os.makedirs(args.out_root, exist_ok=True)
    random.seed(args.seed)

    # =========================
    # 1) ESSAYER MODE LOCAL
    # =========================
    if args.local_train and args.local_valid and args.local_test:
        print("[INFO] Chargement en MODE LOCAL.")
        train_lines = read_lines(args.local_train)
        valid_lines = read_lines(args.local_valid)
        test_lines  = read_lines(args.local_test)
    else:
        # =========================
        # 2) ESSAYER PARQUET HTTP HF
        # =========================
        print("[INFO] Tentative de chargement via Parquet HTTP Hugging Face (sans S3).")
        urls = {
            "train":      "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
            "validation": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/validation-00000-of-00001.parquet",
            "test":       "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet",
        }
        ds = None
        try:
            from datasets import load_dataset
            # IMPORTANT (datasets 1.13.x): quand data_files est un dict {split: url}, ne PAS passer split=...
            ds = load_dataset("parquet", data_files=urls)  # -> DatasetDict avec "train","validation","test"
            print("[OK] Parquet HTTP chargé.")
            train_lines = extract_text_list(ds["train"])
            valid_lines = extract_text_list(ds["validation"])
            test_lines  = extract_text_list(ds["test"])
        except Exception as e_hf_parquet:
            print(f"[WARN] Échec Parquet HTTP HF: {e_hf_parquet}")
            print("[INFO] Tentative de fallback via wikitext-2-raw-v1 (peut utiliser S3).")
            # =========================
            # 3) FALLBACK WIKITEXT CLASSIQUE (peut être bloqué par S3)
            # =========================
            try:
                from datasets import load_dataset
                ds2 = load_dataset("wikitext", "wikitext-2-raw-v1")
                train_lines = [x["text"] for x in ds2["train"] if isinstance(x.get("text",""), str)]
                valid_lines = [x["text"] for x in ds2["validation"] if isinstance(x.get("text",""), str)]
                test_lines  = [x["text"] for x in ds2["test"] if isinstance(x.get("text",""), str)]
            except Exception as e_wt2:
                raise SystemExit(
                    "\n[ERREUR] Impossible de charger WikiText-2 par HTTP Parquet ET par le loader classique.\n"
                    "Solutions :\n"
                    "  - Fournir des fichiers locaux avec --local_train/--local_valid/--local_test\n"
                    "  - Ou réessayer depuis une machine avec accès HTTP à huggingface.co\n"
                    f"Détails:\nHTTP Parquet: {e_hf_parquet}\nWT2 classic: {e_wt2}\n"
                )

    # 4) Nettoyage basique
    def clean(lines):
        out = []
        for s in lines:
            if len(s.strip()) >= args.min_len:
                out.append(s.strip())
        return out
    train_lines, valid_lines, test_lines = map(clean, (train_lines, valid_lines, test_lines))

    # 5) Construire mapping swap
    mapping, pattern = build_gender_maps()

    # 6) Split A/B selon p
    N = len(train_lines)
    NA = int(round(p_value * N))
    idx = list(range(N))
    random.shuffle(idx)
    A_idx = set(idx[:NA])

    envA, envB = [], []
    for i, s in enumerate(train_lines):
        if i in A_idx:
            envA.append(s)
        else:
            envB.append(swap_gender(s, mapping, pattern))

    # 7) Écrire sorties
    tag = f"p{int(round(p_value * 100)):03d}"
    env_dir = os.path.join(args.out_root, f"envs_{tag}")
    erm_dir = os.path.join(args.out_root, f"erm_{tag}")
    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(erm_dir, exist_ok=True)

    def write(path, lines):
        with open(path, "w", encoding="utf-8") as f:
            for s in lines:
                f.write(s + "\n")

    write(os.path.join(env_dir, "envA.txt"), envA)
    write(os.path.join(env_dir, "envB.txt"), envB)
    write(os.path.join(args.out_root, "valid.txt"), valid_lines)
    write(os.path.join(args.out_root, "test.txt"),  test_lines)
    write(os.path.join(erm_dir, "all.txt"), envA + envB)

    print(f"[OK] Env-A: {len(envA)}  Env-B: {len(envB)}  (N={N}, p={p_value})")
    print(f"[OUT] {env_dir}/envA.txt, {env_dir}/envB.txt")
    print(f"[OUT] {args.out_root}/valid.txt, {args.out_root}/test.txt")
    print(f"[OUT] {erm_dir}/all.txt")

if __name__ == "__main__":
    main()
