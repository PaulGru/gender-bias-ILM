#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import subprocess
import re
import math
import torch
import shutil
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizer
from invariant_distilbert import InvariantDistilBertForMaskedLM

# === PARAMETERS ===
generated_root = "generated_splits"
bias_output_root = "bias_outputs_2500"
os.makedirs(bias_output_root, exist_ok=True)

learning_rates = [1e-5, 5e-5]
seeds = [0, 1, 2, 3, 4]
p_values = [0.1, 0.25, 0.5, 0.7, 0.9]
steps_list = [10, 50, 100, 200, 1000, 2500]
methods = ["eLM", "iLM"]
model_type = "distilbert-base-uncased"
use_cuda = torch.cuda.is_available()

# === GENDER PAIRS & DICTIONARY ===
GENDER_PAIRS = [
    ("he", "she"), ("him", "her"), ("his", "hers"),
    ("man", "woman"), ("men", "women"),
    ("boy", "girl"), ("boys", "girls"),
    ("father", "mother"), ("fathers", "mothers"),
    ("son", "daughter"), ("sons", "daughters"),
    ("brother", "sister"), ("brothers", "sisters"),
    ("uncle", "aunt"), ("uncles", "aunts"),
    ("husband", "wife"), ("husbands", "wives"),
    ("actor", "actress"), ("actors", "actresses"),
    ("king", "queen"), ("kings", "queens"),
    ("prince", "princess"), ("princes", "princesses"),
    ("mr.", "mrs."), ("mr", "mrs"),
    ("male", "female"), ("males", "females"),
    ("gentleman", "lady"), ("gentlemen", "ladies"),
    ("boyfriend", "girlfriend"), ("boyfriends", "girlfriends"),
    ("stepfather", "stepmother"), ("stepfathers", "stepmothers"),
    ("spokesman", "spokeswoman"), ("spokesmen", "spokeswomen"),
    ("hero", "heroine"), ("heroes", "heroines"),
    ("grandson", "granddaughter"), ("grandsons", "granddaughters"),
]
GENDER_DICT = {w1.lower(): w2.lower() for w1, w2 in GENDER_PAIRS}
GENDER_DICT.update({w2.lower(): w1.lower() for w1, w2 in GENDER_PAIRS})


def swap_gender_terms(sentence, gender_dict):
    tokens = sentence.split()
    swapped = []
    for token in tokens:
        low = token.lower()
        if low in gender_dict:
            sub = gender_dict[low]
            if token[0].isupper():
                sub = sub.capitalize()
            swapped.append(sub)
        else:
            swapped.append(token)
    return ' '.join(swapped)


def prepare_split(dataset, seed, p_env_a, output_dir):
    """
    Create two environments A (p_env_a fraction of original train) and B (remaining, swapped genders) and
    write to files under output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_lines = [l.strip() for l in dataset['train']['text'] if l.strip()]
    val_lines   = [l.strip() for l in dataset['validation']['text'] if l.strip()]
    test_lines  = [l.strip() for l in dataset['test']['text'] if l.strip()]

    random.seed(seed)
    env_a, env_b = [], []
    for line in tqdm(train_lines, desc=f"Split seed={seed}, p={p_env_a}"):
        if random.random() < p_env_a:
            env_a.append(line)
        else:
            env_b.append(swap_gender_terms(line, GENDER_DICT))

    # write train env
    train_env = os.path.join(output_dir, 'train_env')
    val_env   = os.path.join(output_dir, 'val_env')
    os.makedirs(train_env, exist_ok=True)
    os.makedirs(val_env, exist_ok=True)

    with open(os.path.join(train_env, 'env_A.txt'), 'w') as f:
        f.write("\n".join(env_a))
    with open(os.path.join(train_env, 'env_B.txt'), 'w') as f:
        f.write("\n".join(env_b))
    with open(os.path.join(output_dir, 'all_train.txt'), 'w') as f:
        f.write("\n".join(env_a + env_b))

    # write validation/test
    with open(os.path.join(val_env, 'val_ind.txt'), 'w') as f:
        f.write("\n".join(val_lines))
    with open(os.path.join(val_env, 'test.txt'), 'w') as f:
        f.write("\n".join(test_lines))


def compute_bias(model, tokenizer, test_lines):
    """
    Compute bias scores on test_lines: return list of B_H values.
    """
    scores = []
    model.eval()
    if use_cuda:
        model.to('cuda')

    for line in tqdm(test_lines, desc="Computing bias"):
        words = re.findall(r"\b\w+\b", line.lower())
        found = [w for w in words if w in GENDER_DICT]
        if len(found) != 1:
            continue
        target = found[0]
        opposite = GENDER_DICT[target]
        masked = re.sub(rf"\b{target}\b", tokenizer.mask_token, line, flags=re.IGNORECASE)
        inputs = tokenizer(masked, return_tensors='pt')
        if use_cuda:
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad(): out = model(**inputs)
        mask_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_idx[0]) == 0:
            continue
        logits = out.logits[mask_idx][0]
        probs = logits.softmax(dim=-1)
        try:
            i1 = tokenizer.convert_tokens_to_ids(target)
            i2 = tokenizer.convert_tokens_to_ids(opposite)
            p1 = probs[i1].item()
            p2 = probs[i2].item()
        except:
            continue
        if p1 + p2 < 1e-8:
            continue
        p = p1 / (p1 + p2)
        H = 0 if p in (0,1) else - (p * math.log2(p) + (1-p) * math.log2(1-p))
        scores.append(1 - H)
    return scores


def main():
    # load dataset once
    print("Loading Wikitext-2 dataset...")
    dataset = load_dataset(
        "text",
        data_files={
            "train":      "/home/p.grunenwald/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wiki.train.raw",
            "validation": "/home/p.grunenwald/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wiki.valid.raw",
            "test":       "/home/p.grunenwald/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw",
        },
        split={
            "train": "train",
            "validation": "validation",
            "test": "test"
        }
    )

    for lr in learning_rates:
        for seed in seeds:
            for p in p_values:
                tag = f"split{seed}_p{str(p).replace('.', '')}"
                split_dir = os.path.join(generated_root, tag)
                prepare_split(dataset, seed, p, split_dir)

                test_file = os.path.join(split_dir, 'val_env', 'test.txt')
                if not os.path.exists(test_file):
                    print(f"Missing test file for {tag}, skipping.")
                    continue
                with open(test_file) as f:
                    test_lines = [l.strip() for l in f if l.strip()]

                for method in methods:
                    train_folder = split_dir if method == 'eLM' else os.path.join(split_dir, 'train_env')
                    for steps in steps_list:

                        model_path = os.path.join('model', f"{tag}_lr{str(lr).replace('.', '')}_steps{steps}", method)
                        os.makedirs(model_path, exist_ok=True)

                        if not os.path.isdir(model_path) or not os.listdir(model_path):
                        
                            cmd = (
                                f"python3 run_invariant_mlm.py "
                                f"--model_name_or_path {model_type} "
                                f"--train_file {train_folder} "
                                f"--validation_file {os.path.join(split_dir, 'val_env', 'val_ind.txt')} "
                                f"--do_train --do_eval --nb_steps {steps} --learning_rate {lr} "
                                f"--output_dir {model_path} "
                                f"--seed {seed} --per_device_train_batch_size 8 "
                                f"--preprocessing_num_workers 4 --gradient_accumulation_steps 2 "
                                f"--use_auth_token "
                                f"--fp16 --overwrite_output_dir"
                            )
                            print(f"Training {method} on {tag} (lr={lr}, seed={seed})...")
                            subprocess.run(cmd, shell=True, check=True)

                        tokenizer = DistilBertTokenizer.from_pretrained(
                            model_path,
                            use_auth_token=True
                        )
                        model = InvariantDistilBertForMaskedLM.from_pretrained(
                            model_path,
                            use_auth_token=True
                        )
                        bias_scores = compute_bias(model, tokenizer, test_lines)
                        df = pd.DataFrame({'bias': bias_scores})
                        out_csv = os.path.join(bias_output_root, f"{tag}_lr{str(lr).replace('.', '')}_steps{steps}_{method}.csv")
                        df.to_csv(out_csv, index=False)
                        print(f"Saved bias scores to {out_csv}")

                        # Remove saved model to free disk space
                        if os.path.isdir(model_path):
                            shutil.rmtree(model_path)
                            print(f"Removed model directory {model_path} to free up disk space.")

if __name__ == '__main__':
    main()
