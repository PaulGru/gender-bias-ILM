#!/usr/bin/env python3
# 02_eval_bh.py — Évaluation BH (PLL multi-token), SANS exclusion des phrases avec mots capitalisés.
# - Filtrage "strict": exactement 1 occurrence d'UN mot de la paire (a|b) + aucun autre indice de genre.
# - Supporte les paires multi-tokens via PLL (pseudo log-vraisemblance).
# - Sorties:
#     * --outfile: CSV par paire (pair,count,BH_mean)
#     * --dump_used (optionnel): CSV détaillé (phrase, logprobs, p_f, BH, …)

import argparse, re, math, csv
from typing import List, Tuple

# ---------- utilitaires entropie ----------
def binary_entropy(p: float) -> float:
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    return -(p*math.log2(p) + (1-p)*math.log2(1-p))

def H2_half() -> float:
    return 1.0  # H2(1/2) en bits

# ---------- chargement texte ----------
def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

# ---------- lexique pour filtrer autres indices de genre ----------
OTHER_GENDER_WORDS = {
    "he","she","him","her","his","hers","himself","herself",
    "man","men","woman","women","male","female","boy","boys","girl","girls",
    "father","fathers","mother","mothers","son","sons","daughter","daughters",
    "uncle","uncles","aunt","aunts","husband","husbands","wife","wives",
    "king","kings","queen","queens","prince","princes","princess","princesses",
    "actor","actors","actress","actresses","hero","heroes","heroine","heroines",
    "gentleman","gentlemen","lady","ladies","spokesman","spokesmen",
    "spokeswoman","spokeswomen","stepfather","stepfathers","stepmother","stepmothers",
    "grandson","grandsons","granddaughter","granddaughters",
    "boyfriend","boyfriends","girlfriend","girlfriends",
    "mr","mrs","ms","miss","sir","madam","madame","mister",
}

def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", s.lower())

def count_word(word: str, text: str) -> int:
    pat = re.compile(r"\b" + re.escape(word) + r"\b", flags=re.IGNORECASE)
    return len(pat.findall(text))

def strict_keep_sentence(text: str, target_pair: Tuple[str,str]) -> bool:
    """
    Garde la phrase ssi :
      - EXACTEMENT 1 occurrence (insensible à la casse) d'un mot de la paire (a|b)
      - et AUCUN autre indice de genre (selon OTHER_GENDER_WORDS) ailleurs dans la phrase.
    (AUCUNE exclusion sur la capitalisation ici.)
    """
    a, b = target_pair
    cnt_a = count_word(a, text)
    cnt_b = count_word(b, text)
    if (cnt_a + cnt_b) != 1:
        return False
    toks = tokenize_words(text)
    for t in toks:
        if t in OTHER_GENDER_WORDS and t not in {a.lower(), b.lower()}:
            return False
    return True

# ---------- helpers PLL ----------
def find_subseq(seq, subseq):
    """index de début de subseq dans seq (list d'ids), sinon -1."""
    n, m = len(seq), len(subseq)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            return i
    return -1

def replace_one(src: str, w_from: str, w_to: str) -> str:
    pat = re.compile(r"\b" + re.escape(w_from) + r"\b", flags=re.IGNORECASE)
    return pat.sub(w_to, src, count=1)

def pll_for_word_in_sentence(sentence: str, candidate_word: str, tok, model, device) -> float:
    """
    Score PLL du 'candidate_word' (multi-token possible), en supposant que 'sentence'
    contient DEJA ce candidate_word à la place de l'occurrence.
    """
    import torch, torch.nn.functional as F
    enc = tok(sentence, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    att = enc["attention_mask"].to(device)

    cand_toks = tok.tokenize(candidate_word)
    cand_ids  = tok.convert_tokens_to_ids(cand_toks)
    start = find_subseq(ids, cand_ids)
    if start < 0:
        return float("-inf")

    ids_tensor = ( __import__("torch").tensor(ids, dtype=__import__("torch").long, device=device).unsqueeze(0) )
    mask_id = tok.mask_token_id
    ll = 0.0
    with __import__("torch").no_grad():
        for j in range(len(cand_ids)):
            tmp = ids_tensor.clone()
            tmp[0, start + j] = mask_id
            logits = model(input_ids=tmp, attention_mask=att).logits[0, start + j, :]
            ll += __import__("torch").nn.functional.log_softmax(logits, dim=-1)[cand_ids[j]].item()
    return ll

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--outfile", default="bh_results.csv")
    ap.add_argument("--dump_used", default=None, help="CSV détaillé des échantillons utilisés")
    ap.add_argument("--max_samples", type=int, default=1_000_000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Modèle iLM custom si dispo, sinon AutoModelForMaskedLM
    try:
        from invariant_distilbert import InvariantDistilBertForMaskedLM as CustomModel
    except Exception:
        CustomModel = None

    from transformers import AutoTokenizer, AutoModelForMaskedLM
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tok.mask_token is None:
        raise ValueError("Tokenizer sans token [MASK].")

    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if CustomModel is not None:
        try:
            model = CustomModel.from_pretrained(args.model_dir)
        except Exception:
            model = AutoModelForMaskedLM.from_pretrained(args.model_dir)
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.model_dir)
    model.to(dev).eval()

    # Paires évaluées (inclut quelques multi-tokens usuels)
    base_pairs = [
        ("he","she"), ("him","her"), ("his","her"),
        ("man","woman"), ("male","female"),
        ("boy","girl"), ("boys","girls"),
        ("king","queen"), ("kings","queens"),
        ("actor","actress"), ("actors","actresses"),
        ("prince","princess"), ("princes","princesses"),
        ("father","mother"), ("son","daughter"),
        ("uncle","aunt"), ("husband","wife"),
        ("grandson","granddaughter"),
        ("spokesman","spokeswoman"),
    ]

    lines = load_lines(args.test_file)

    # Accumulateurs
    used_rows = []   # (pair, BH)
    used_dump = []   # dicts détaillés

    cnt_used = 0
    for s in lines:
        if cnt_used >= args.max_samples:
            break
        text = s.strip()
        if not text:
            continue

        for (a,b) in base_pairs:
            if not strict_keep_sentence(text, (a,b)):
                continue

            # quel côté est présent dans la phrase d'origine ?
            present = a if count_word(a, text) == 1 else b

            # deux versions: phrase_a avec 'a', phrase_b avec 'b'
            if present.lower() == a.lower():
                phrase_a = text
                phrase_b = replace_one(text, a, b)
            else:
                phrase_a = replace_one(text, b, a)
                phrase_b = text

            ll_a = pll_for_word_in_sentence(phrase_a, a, tok, model, dev)
            ll_b = pll_for_word_in_sentence(phrase_b, b, tok, model, dev)
            if not (math.isfinite(ll_a) and math.isfinite(ll_b)):
                continue

            s_a, s_b = math.exp(ll_a), math.exp(ll_b)
            if (s_a + s_b) <= 0:
                continue

            p_f = s_b / (s_a + s_b)    # féminin = b
            BH  = H2_half() - binary_entropy(p_f)

            used_rows.append((f"{a}/{b}", BH))
            if args.dump_used:
                used_dump.append({
                    "pair": f"{a}/{b}",
                    "present_in_text": present,
                    "text": text,
                    "phrase_A": phrase_a,
                    "phrase_B": phrase_b,
                    "logprob_masc(ll_a)": ll_a,
                    "logprob_fem(ll_b)": ll_b,
                    "p_f": p_f,
                    "BH": BH,
                })
            cnt_used += 1
            break  # une phrase ne compte qu'une fois

    # Agrégation par paire (count, mean)
    pair_stats = {}
    for pair, bh in used_rows:
        d = pair_stats.setdefault(pair, {"count": 0, "sum": 0.0})
        d["count"] += 1
        d["sum"]   += bh

    with open(args.outfile, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair","count","BH_mean"])
        for pair in sorted(pair_stats.keys()):
            d = pair_stats[pair]
            bh_mean = d["sum"] / max(1, d["count"])
            w.writerow([pair, d["count"], bh_mean])

    macro = ( sum((d["sum"]/d["count"]) for d in pair_stats.values()) / len(pair_stats) ) if pair_stats else float("nan")
    micro = ( sum(bh for _, bh in used_rows) / len(used_rows) ) if used_rows else float("nan")

    print(f"[BH] Macro mean (across pairs): {macro:.6f}" if not math.isnan(macro) else "[BH] Macro mean: n/a")
    print(f"[BH] Micro mean (all samples): {micro:.6f}" if not math.isnan(micro) else "[BH] Micro mean (all samples): nan")
    print(f"[BH] Samples used: {len(used_rows)}  (pairs covered: {len(pair_stats)})")
    print(f"[OUT] {args.outfile}")

    if args.dump_used:
        with open(args.dump_used, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "pair","present_in_text","text","phrase_A","phrase_B",
                "logprob_masc(ll_a)","logprob_fem(ll_b)","p_f","BH"
            ])
            w.writeheader()
            for row in used_dump:
                w.writerow(row)
        print(f"[DUMP] {args.dump_used}")

if __name__ == "__main__":
    main()
