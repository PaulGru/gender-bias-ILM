# pip install inflect
import inflect

# 1. Définir vos paires de base (toujours en minuscules, forme singulier)
base_pairs = [
    ("actor",     "actress"),
    ("boy",       "girl"),
    ("boyfriend", "girlfriend"),
    ("father",    "mother"),
    ("gentleman", "lady"),
    ("grandson",  "granddaughter"),
    ("he",        "she"),
    ("hero",      "heroine"),
    ("him",       "her"),
    ("husband",   "wife"),
    ("king",      "queen"),
    ("male",      "female"),
    ("man",       "woman"),
    ("mr.",       "mrs."),
    ("prince",    "princess"),
    ("son",       "daughter"),
    ("spokesman", "spokeswoman"),
    ("stepfather","stepmother"),
    ("uncle",     "aunt"),
]

# 2. Initialiser l'outil de flexion (pour pluriel)
p = inflect.engine()

# 3. Construire le dictionnaire de toutes les variantes
gender_map = {}

for masc, fem in base_pairs:
    # formes à générer : singulier/pluriel × lower/title/upper
    forms = []
    masc_plur = p.plural(masc)
    fem_plur  = p.plural(fem)
    
    # combinaisons
    for (m_base, f_base) in [(masc, fem), (masc_plur, fem_plur)]:
        for transform in (str.lower, str.title, str.upper):
            m_var = transform(m_base)
            f_var = transform(f_base)
            forms.append((m_var, f_var))
    
    # remplir gender_map (bidirectionnel)
    for m_var, f_var in forms:
        gender_map[m_var] = f_var
        gender_map[f_var] = m_var

# 4. Exemples d’utilisation
tests = ["actor", "Actress", "ACTORS", "girls", "He", "king", "QUEENS"]
for word in tests:
    opposite = gender_map.get(word)
    print(f"{word:8} → {opposite}")
