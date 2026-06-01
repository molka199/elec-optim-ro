# Optimisation d'un Réseau de Distribution d'Électricité

**Projet de Recherche Opérationnelle — ENIT, Tunis | Mars 2025**

Modélisation et résolution d'un problème d'optimisation des coûts de production et de transport dans un réseau reliant **6 centrales électriques** à **8 villes**, sous contraintes de capacité et de pertes d'énergie.

---

## Problème

**Variables de décision :** $x_{ij}$ = énergie produite par la centrale $i$ et acheminée vers la ville $j$ (MW)

**Fonction objectif :**

$$\min \sum_{i=1}^{6} \sum_{j=1}^{8} (c^{\text{prod}}_i + c^{\text{trans}}_{ij}) \cdot x_{ij}$$

**Contraintes :**

| # | Description | Expression |
|---|-------------|------------|
| C1 | Satisfaction de la demande | $\sum_i (1 - p_{ij}) \cdot x_{ij} \geq d_j \quad \forall j$ |
| C2 | Capacité des centrales | $\sum_j x_{ij} \leq \text{CAP}_i \quad \forall i$ |
| C3 | Capacité des lignes | $x_{ij} \leq L_{ij} \quad \forall i,j$ |
| C4 | Non-négativité | $x_{ij} \geq 0$ |

Avec $p_{ij}$ les pertes d'énergie sur la ligne $i \to j$ (1.5% à 4.2%).

---

## Données

| Tableau | Contenu |
|---------|---------|
| Centrales | 6 centrales, capacités 70–110 MW, coûts 27–32 €/MW |
| Villes | 8 villes, demandes 35–65 MW (total : 400 MW) |
| Transport | Coûts 4–9 €/MW par liaison |
| Pertes | 1.5% à 4.2% par liaison |
| Lignes | Capacités 50–90 MW par liaison |

---

## Structure du projet

```
elec_optim/
├── Data_Sujet_01.xlsx              # Données du problème
├── main.py                         # Script principal – lance & compare tous les algo
├── utils/
│   └── data_loader.py              # Chargement données, calcul coût, vérif contraintes
├── exact/
│   └── pl_solver.py                # Résolution exacte (Programmation Linéaire, PuLP/CBC)
├── heuristiques/
│   └── gloutonne.py                # Heuristique gloutonne + réallocation post-opt
├── metaheuristiques/
│   ├── tabou.py                    # Recherche Tabou (voisinage par transfert partiel)
│   ├── recuit_simule.py            # Recuit Simulé (SA)
│   ├── fourmis.py                  # Colonie de Fourmis (ACO – Ant Colony System)
│   └── genetique.py                # Algorithme Génétique (AG)
└── results/
    └── comparaison.csv             # Tableau comparatif généré automatiquement
```

---

## Algorithmes implémentés

### Résolution exacte
- **Programmation Linéaire (PuLP/CBC)** — donne la solution optimale certifiée.

### Heuristique
- **Gloutonne** — tri des paires (centrale, ville) par coût effectif $= (c^\text{prod}_i + c^\text{trans}_{ij}) / (1-p_{ij})$, puis post-optimisation par réallocation intelligente.

### Métaheuristiques
| Algorithme | Principe clé | Paramètres principaux |
|------------|-------------|----------------------|
| **Recherche Tabou** | Voisinage par transfert partiel avec compensation des pertes, liste tabou sur les mouvements, critère d'aspiration, diversification par redémarrage | `max_iter=500`, `taille_tabou=20` |
| **Recuit Simulé** | Acceptation probabiliste $e^{-\Delta/T}$, refroidissement géométrique $T \leftarrow \alpha T$ | `T_init=5000`, `alpha=0.995` |
| **Colonie de Fourmis (ACO)** | Phéromone sur les liaisons utilisées, heuristique $\eta=1/\text{coût\_effectif}$, dépôt élitiste | `n_fourmis=20`, `n_iter=150` |
| **Algorithme Génétique** | Croisement uniforme par ville, réparation des individus infaisables, élitisme | `pop=60`, `n_gen=200` |

---

## Résultats

| Algorithme | Coût total (€) | Écart / optimum | Faisable | Temps (s) |
|------------|---------------|-----------------|----------|-----------|
| PL Exacte (PuLP) | **13 690.79** | 0.00% | ✓ | ~0.01 |
| Colonie de Fourmis | 13 694.92 | +0.03% | ✓ | ~0.72 |
| Recuit Simulé | 13 716.24 | +0.19% | ✓ | ~1.94 |
| Algorithme Génétique | 13 752.99 | +0.45% | ✓ | ~2.52 |
| Recherche Tabou | 13 816.01 | +0.91% | ✓ | ~0.80 |
| Heuristique Gloutonne | 13 816.33 | +0.92% | ✓ | <0.001 |

> **L'ACO obtient 13 694.92 €, soit à seulement 0.03% de l'optimum exact**, en un temps inférieur à 1 seconde.

---

## Installation & utilisation

```bash
# Dépendances
pip install pulp numpy pandas openpyxl

# Lancer tous les algorithmes et afficher le tableau comparatif
python main.py

# Sans la résolution exacte (si PuLP non disponible)
python main.py --no-exact

# Lancer un algorithme individuellement
python exact/pl_solver.py
python heuristiques/gloutonne.py
python metaheuristiques/tabou.py
python metaheuristiques/recuit_simule.py
python metaheuristiques/fourmis.py
python metaheuristiques/genetique.py
```

---

## Technologies

`Python 3.10+` · `PuLP` · `NumPy` · `Pandas` · `openpyxl`

---

*ENIT – École Nationale d'Ingénieurs de Tunis, Département Génie Industriel*
