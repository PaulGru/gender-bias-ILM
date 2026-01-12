#!/usr/bin/env python3
# 06_make_perplexity_means.py (updated)
# Calcule les moyennes/écarts de perplexité par mode à partir d'un dossier de fichiers *_eval_results.txt
# Compatible avec ERM, iLM, IRM-Games et IRM-G φ fixed.

import os, sys, glob, re, math

def main(folder: str, out_csv: str = None, modes: str = None):
    pat = re.compile(r'^(erm|ilm|game|game_phi_fixed)_.+_eval_results\.txt$')
    name_map = {'erm': 'eLM', 'ilm': 'iLM', 'game':'IRM-Games', 'game_phi_fixed':'IRM-G φ fixed'}

    keep = {m.strip().lower() for m in modes.split(",")} if modes else None

    values = {}  # mode -> list of perplexities
    for fp in glob.glob(os.path.join(folder, '*.txt')):
        fn = os.path.basename(fp)
        m = pat.match(fn)
        if not m:
            continue
        mode = m.group(1)
        if keep and mode not in keep:
            continue
        perp = None
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("perplexity"):
                    mm = re.search(r'([0-9]+(\.[0-9]+)?)', line)
                    if mm:
                        try:
                            perp = float(mm.group(1))
                        except Exception:
                            perp = None
                    break
        if perp is not None:
            values.setdefault(mode, []).append(perp)

    rows = []
    for mode, arr in values.items():
        n = len(arr)
        mean = sum(arr)/n if n>0 else float('nan')
        variance = sum((x-mean)**2 for x in arr)/(n-1) if n>1 else 0.0
        std = math.sqrt(variance)
        se = std / math.sqrt(n) if n>0 else float('nan')
        ci95 = 1.96 * se if n>0 else float('nan')
        rows.append((name_map.get(mode,mode), mean, std, n, se, ci95))

    order = ['eLM','iLM','IRM-Games','IRM-G φ fixed']
    rows.sort(key=lambda r: (order.index(r[0]) if r[0] in order else 99))

    header = "model,mean,std,n,se,ci95"
    lines = [header] + [f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}" for r in rows]
    csv_text = "\n".join(lines)
    if out_csv:
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write(csv_text + "\n")
        print(f"[OUT] {out_csv}")
    else:
        print(csv_text)

if __name__ == "__main__":
    # usage:
    #   python 06_make_perplexity_means.py eval_results_perplexity > perplexity_by_mode.csv
    #   python 06_make_perplexity_means.py eval_results_perplexity perplexity_by_mode.csv
    #   python 06_make_perplexity_means.py eval_results_perplexity perplexity_by_mode.csv "ilm,game_phi_fixed"
    folder = sys.argv[1] if len(sys.argv) > 1 else "eval_results_perplexity"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    modes = sys.argv[3] if len(sys.argv) > 3 else None
    main(folder, out, modes)
