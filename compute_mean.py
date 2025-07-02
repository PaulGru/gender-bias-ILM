#!/usr/bin/env python3
"""
Parcours un dossier contenant des fichiers CSV (avec un en-tête 'bias'),
calcule la moyenne de chaque fichier, puis écrit dans un nouveau dossier
un CSV ne contenant que la valeur moyenne pour chaque fichier avec le même nom.
"""
import os
import pandas as pd

def compute_means(input_dir: str, output_dir: str) -> None:
    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Parcours de tous les fichiers dans input_dir
    for entry in os.listdir(input_dir):
        if not entry.lower().endswith('.csv'):
            continue
        input_path = os.path.join(input_dir, entry)
        # Lecture du CSV avec header (première ligne contenant 'bias')
        df = pd.read_csv(input_path)
        # Calcul de la moyenne de la colonne 'bias' (ou la première colonne si le header diffère)
        if 'bias' in df.columns:
            mean_val = df['bias'].mean()
        else:
            mean_val = df.iloc[:, 0].mean()
        # Préparation du DataFrame de sortie (une seule valeur)
        out_df = pd.DataFrame([mean_val], columns=['bias'])
        # Chemin du fichier de sortie
        output_path = os.path.join(output_dir, entry)
        # Écriture sans index
        out_df.to_csv(output_path, index=False)
        print(f"Traité {entry} → moyenne = {mean_val:.6f}")

if __name__ == '__main__':
    # Dossier source contenant les CSV originaux
    input_folder = 'bias_outputs_xxx'
    # Dossier de destination pour les CSV des moyennes
    output_folder = 'bias_outputs_xxx_mean'
    compute_means(input_folder, output_folder)
    print(f"Terminé ! Fichiers écrits dans '{output_folder}'")
