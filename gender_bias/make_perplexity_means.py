import os, sys, glob, re

def main(folder: str, out_csv: str = None):
    pat = re.compile(r'^(erm|ilm)_.+_eval_results\.txt$')
    name_map = {'erm': 'eLM', 'ilm': 'iLM'}
    values = {'erm': [], 'ilm': []}

    for fp in glob.glob(os.path.join(folder, '*.txt')):
        fn = os.path.basename(fp)
        m = pat.match(fn)
        if not m:
            continue
        mode = m.group(1)
        perp = None
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if 'perplexity' in line:
                    try:
                        perp = float(line.split('=')[1].strip())
                    except Exception:
                        perp = None
                    break
        if perp is not None:
            values[mode].append(perp)

    # prépare le CSV
    lines = ["model,average_perplexity,n"]
    for mode in ('erm', 'ilm'):
        vals = values[mode]
        if vals:
            avg = sum(vals) / len(vals)
            lines.append(f"{name_map[mode]},{avg:.6f},{len(vals)}")

    csv_text = "\n".join(lines)
    if out_csv:
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write(csv_text + "\n")
        print(f"[OUT] {out_csv}")
    else:
        print(csv_text)

if __name__ == "__main__":
    # usage:
    #   python make_perplexity_means.py eval_results_perplexity > perplexity_by_mode.csv
    #   python make_perplexity_means.py eval_results_perplexity perplexity_by_mode.csv
    folder = sys.argv[1] if len(sys.argv) > 1 else "eval_results_perplexity"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    main(folder, out)
