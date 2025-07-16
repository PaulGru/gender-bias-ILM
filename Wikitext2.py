import os
import random
import subprocess
import re
import math
import torch
import shutil
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizer
from invariant_distilbert import InvariantDistilBertForMaskedLM
from huggingface_hub import hf_hub_download
import inflect

# === PARAMETERS ===
generated_root = "generated_splits"
bias_output_root = "output_bias_scores"
os.makedirs(generated_root, exist_ok=True)
os.makedirs(bias_output_root, exist_ok=True)

learning_rates = [5e-5]
seeds = [0, 1, 2, 3, 4]
# Tailles relatives de B par rapport à A en pourcentage
rel_sizes = [10, 25, 30, 50, 70, 75, 90, 100]
steps_list = [10, 50, 100, 200, 1000, 2500]
methods = ["eLM", "iLM"]

# === GENDER PAIRS & DICTIONARY DYNAMIQUE ===
# Paires de base (minuscules, singulier)
base_pairs = [
    ("actor", "actress"),
    ("boy", "girl"),
    ("boyfriend", "girlfriend"),
    ("father", "mother"),
    ("gentleman", "lady"),
    ("grandson", "granddaughter"),
    ("he", "she"),
    ("hero", "heroine"),
    ("him", "her"),
    ("husband", "wife"),
    ("king", "queen"),
    ("male", "female"),
    ("man", "woman"),
    ("mr.", "mrs."),
    ("prince", "princess"),
    ("son", "daughter"),
    ("spokesman", "spokeswoman"),
    ("stepfather", "stepmother"),
    ("uncle", "aunt")
]

# Initialiser l'outil inflect pour les pluriels
p = inflect.engine()

# Construire le dictionnaire de variantes (singulier/pluriel × lower/title/upper)
GENDER_DICT = {}
for masc, fem in base_pairs:
    masc_plur = p.plural(masc)
    fem_plur = p.plural(fem)
    for (m_base, f_base) in [(masc, fem), (masc_plur, fem_plur)]:
        for transform in (str.lower, str.title, str.upper):
            m_var = transform(m_base)
            f_var = transform(f_base)
            GENDER_DICT[m_var] = f_var
            GENDER_DICT[f_var] = m_var


def swap_gender_terms(sentence, gender_dict):
    # Découpe en tokens : mots ou ponctuation
    tokens = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
    swapped = []
    for token in tokens:
        swapped.append(gender_dict.get(token, token))
    # Reconstruire la phrase avec espaces intelligents
    result = ""
    for i, tok in enumerate(swapped):
        if i > 0 and re.match(r"\w", tok) and re.match(r"\w", swapped[i-1]):
            result += " "
        result += tok
    return result


def prepare_split(dataset, seed, p_env_a, output_dir):
    """
    Create two environments A (p_env_a fraction of original train) and B (remaining, swapped genders)
    and write to files under output_dir.
    """
    # create output directories
    train_env = os.path.join(output_dir, 'train_env')
    val_env = os.path.join(output_dir, 'val_env')
    os.makedirs(train_env, exist_ok=True)
    os.makedirs(val_env, exist_ok=True)

    # load lines
    train_lines = [l.strip() for l in dataset['train']['text'] if l.strip()]
    val_lines   = [l.strip() for l in dataset['validation']['text'] if l.strip()]
    test_lines  = [l.strip() for l in dataset['test']['text'] if l.strip()]

    # deterministic split by slicing after shuffle
    random.seed(seed)
    random.shuffle(train_lines)
    n = len(train_lines)
    n_a = int(p_env_a * n)
    env_a = train_lines[:n_a]
    env_b = [swap_gender_terms(l, GENDER_DICT) for l in train_lines[n_a:]]
    # env_a = [swap_gender_terms(l, GENDER_DICT) for l in train_lines[:n_a]]
    # env_b = train_lines[n_a:]

    # write train environments
    with open(os.path.join(train_env, 'env_A.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(env_a))
    with open(os.path.join(train_env, 'env_B.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(env_b))

    all_train = env_a + env_b
    random.shuffle(all_train)
    # write combined train file
    with open(os.path.join(output_dir, 'all_train.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(all_train))

    # write validation and test
    with open(os.path.join(val_env, 'val_ind.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(val_lines))
    with open(os.path.join(val_env, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(test_lines))


def compute_bias(model, tokenizer, test_lines, gender_dict, block_size=None, verbose=True):
    """
    Compute bias scores on test_lines: return list of B_H values.
    """
    scores = []
    clean_lines = []
    masked_targets = []
    model.eval()
    model.to('cuda')
    
    # Réplication exacte de la logique run_invariant_mlm.py
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            block_size = 1024

    for line in tqdm(test_lines, desc="Computing bias"):
        words = re.findall(r"\b\w+\b", line)
        found = [w for w in words if w in gender_dict]
        if len(found) != 1:
            continue
        target = found[0]
        opposite = gender_dict[target]
        masked = re.sub(rf"\b{target}\b", tokenizer.mask_token, line, flags=re.IGNORECASE)
        clean_lines.append(masked)
        masked_targets.append((target, opposite))

    if len(clean_lines) == 0:
        if verbose:
            print("[WARN] Aucune phrase valide trouvée pour l'évaluation du biais.")
        return []
    
    # Étape 2 — Créer un dataset HuggingFace
    raw_dataset = Dataset.from_dict({"text": clean_lines})

    # Étape 3 — Tokenize
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            return_special_tokens_mask=True
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=not False,
    )

    # Étape 4 — Group texts à la manière de run_invariant_mlm.py
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [v[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, v in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=16,
        load_from_cache_file=not False,
    )

    # Étape 5 — Évaluation manuelle
    for i, example in enumerate(tqdm(grouped_dataset, desc="Computing bias")):

        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0)
        
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        
        mask_idx = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_idx[0]) == 0:
            continue

        logits = out.logits[0, mask_idx[0][0]]
        probs = logits.softmax(dim=-1)

        # Récupérer les bons mots cibles
        try:
            target, opposite = masked_targets[i]
        except IndexError:
            continue

        toks_target = tokenizer.tokenize(target)
        toks_opposite = tokenizer.tokenize(opposite)
        
        if len(toks_target) != 1 or len(toks_opposite) != 1:
            continue

        try:
            i1 = tokenizer.convert_tokens_to_ids(toks_target[0])
            i2 = tokenizer.convert_tokens_to_ids(toks_opposite[0])
            
            p1 = probs[i1].item()
            p2 = probs[i2].item()
        except:
            continue

        if p1 + p2 < 1e-8:
            continue
        p = p1 / (p1 + p2)
        H = 0 if p in (0, 1) else - (p * math.log2(p) + (1 - p) * math.log2(1 - p))
        scores.append(1 - H)
    
    if verbose:
        print(f"[INFO] Phrases retenues : {len(scores)} / {len(test_lines)}")
        if scores:
            print(f"[INFO] Biais moyen (1 - H) : {sum(scores) / len(scores):.4f}")
            
    return scores


def main():
    # --- PRÉ-TÉLÉCHARGEMENT DES POIDS ---
    # hf_hub_download(
    #     repo_id="distilbert-base-uncased",
    #     filename="pytorch_model.bin",
    #     force_download=False,    # n’écrase pas le cache si déjà présent
    #     resume_download=True     # reprend un téléchargement partiel
    # )
    # → ça garantira qu’au moment du `--model_name_or_path distilbert-base-uncased`
    #   les fichiers sont déjà dans le cache, et n’iront plus frapper l’endpoint
    #   qui renvoie 503.

    # load dataset once
    # print("Loading Wikitext-2 dataset...")
    # dataset = load_dataset(
    #     "text",
    #     data_files={
    #         "train": "/home/p.grunenwald/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wiki.train.raw",
    #         "validation": "/home/p.grunenwald/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wiki.valid.raw",
    #         "test": "/home/p.grunenwald/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw",
    #     },
    #     split={"train": "train", "validation": "validation", "test": "test"}
    # )

    print("Loading Wikitext-2 dataset…")

    urls = {
        "train":      "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "validation": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/validation-00000-of-00001.parquet",
        "test":       "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet",
    }

    dataset = load_dataset(
        "parquet",
        data_files=urls,
        split   = {"train": "train", "validation": "validation", "test": "test"},
    )


    for lr in learning_rates:
        for seed in seeds:
            for r in rel_sizes:
                p_env_a = 1.0 / (1.0 + r/100.0)
                tag = f"split{seed}_r{r}"
                split_dir = os.path.join(generated_root, tag)
                prepare_split(dataset, seed, p_env_a, split_dir)

                test_file = os.path.join(split_dir, 'val_env', 'test.txt')
                if not os.path.exists(test_file):
                    print(f"Missing test file for {tag}, skipping.")
                    continue
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_lines = [l.strip() for l in f if l.strip()]

                for method in methods:
                    train_folder = split_dir if method == 'eLM' else os.path.join(split_dir, 'train_env')
                    for steps in steps_list:
                        model_path = os.path.join(
                            'model', f"{tag}_lr{str(lr).replace('.', '')}_steps{steps}_{method}"
                        )
                        os.makedirs(model_path, exist_ok=True)

                        # entraînement si nécessaire
                        if not os.listdir(model_path):
                            cmd = (
                                f"python3 run_invariant_mlm.py "
                                f"--model_name_or_path distilbert-base-uncased "
                                f"--do_train --train_file {train_folder} "
                                f"--do_eval --validation_file {os.path.join(split_dir, 'val_env', 'val_ind.txt')} "
                                f"--nb_steps {steps} --learning_rate {lr} "
                                f"--output_dir {model_path} --overwrite_output_dir --preprocessing_num_workers 16 "
                                f"--seed {seed} --per_device_train_batch_size 64 --gradient_accumulation_steps 4 "
                                f"--max_seq_length 128 "
                                f"--fp16 "
                            )
                            print(f"Training {method} on {tag} (lr={lr}, seed={seed})...")
                            subprocess.run(cmd, shell=True, check=True)

                        # évaluation du biais
                        tokenizer = DistilBertTokenizer.from_pretrained(
                            model_path,
                            force_download=False,
                            resume_download=False
                        )
                        model = InvariantDistilBertForMaskedLM.from_pretrained(
                            model_path,
                            force_download=False,
                            resume_download=False
                        )
                        
                        model.to('cuda')
                        
                        bias_scores = compute_bias(
                            model=model,
                            tokenizer=tokenizer,
                            test_lines=test_lines,
                            gender_dict=GENDER_DICT,
                            block_size=128
                        )
                        df = pd.DataFrame({'bias': bias_scores})
                        out_csv = os.path.join(
                            bias_output_root,
                            f"{tag}_lr{str(lr).replace('.', '')}_steps{steps}_{method}.csv"
                        )
                        df.to_csv(out_csv, index=False)
                        print(f"Saved bias scores to {out_csv}")

                        # Remove saved model to free disk space
                        shutil.rmtree(model_path)
                        

if __name__ == '__main__':
    main()
