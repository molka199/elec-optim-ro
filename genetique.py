
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import charger_donnees, cout_total, afficher_solution
from heuristiques.gloutonne import _solution_initiale_gloutonne


def _reparer(x: np.ndarray, d: dict) -> np.ndarray:
    """Répare une solution non-faisable en complétant la demande manquante."""
    nc, nv = d['n_centrales'], d['n_villes']
    x = np.clip(x, 0, None)
    # Respecter les capacités des lignes
    x = np.minimum(x, d['capacites_lignes'])
    # Respecter les capacités des centrales (scale down si nécessaire)
    for i in range(nc):
        total_i = np.sum(x[i])
        if total_i > d['capacites_max'][i] + 1e-9:
            x[i] *= d['capacites_max'][i] / total_i
    # Compléter la demande insatisfaite goulûment
    cap_rest = d['capacites_max'] - np.sum(x, axis=1)
    cap_rest = np.maximum(cap_rest, 0)
    livraison = np.sum((1 - d['pertes_energie']) * x, axis=0)

    for j in range(nv):
        deficit = d['demandes_villes'][j] - livraison[j]
        if deficit <= 1e-6:
            continue
        ordre = np.argsort(d['cout_effectif'][:, j])
        for i in ordre:
            if cap_rest[i] < 1e-6:
                continue
            cap_ligne = d['capacites_lignes'][i][j] - x[i][j]
            if cap_ligne < 1e-6:
                continue
            besoin = deficit / (1 - d['pertes_energie'][i][j])
            alloc = min(besoin, cap_rest[i], cap_ligne)
            x[i][j] += alloc
            cap_rest[i] -= alloc
            deficit -= alloc * (1 - d['pertes_energie'][i][j])
            if deficit <= 1e-6:
                break

    return x


def _init_population(pop_size: int, d: dict, rng) -> list:
    pop = []
    # 1 individu = solution gloutonne pure
    pop.append(_solution_initiale_gloutonne(d))
    # Reste = gloutonne + perturbation aléatoire
    while len(pop) < pop_size:
        x = _solution_initiale_gloutonne(d).copy()
        # Perturbation
        nc, nv = d['n_centrales'], d['n_villes']
        n_pert = rng.integers(3, 10)
        for _ in range(n_pert):
            i = rng.integers(nc)
            j = rng.integers(nv)
            k = rng.integers(nc)
            if k != i and x[i][j] > 1e-6:
                delta = rng.uniform(0, x[i][j] * 0.5)
                x[i][j] -= delta
                x[k][j] += delta
        pop.append(_reparer(x, d))
    return pop

def _tournoi(pop: list, couts: np.ndarray, k: int, rng) -> np.ndarray:
    idx = rng.choice(len(pop), size=k, replace=False)
    best = idx[np.argmin(couts[idx])]
    return pop[best].copy()

def _croiser(p1: np.ndarray, p2: np.ndarray, d: dict, rng) -> np.ndarray:
    nv = d['n_villes']
    masque = rng.random(nv) < 0.5
    enfant = np.where(masque[np.newaxis, :], p1, p2)
    return _reparer(enfant, d)

def _muter(x: np.ndarray, d: dict, rng, taux: float = 0.15) -> np.ndarray:
    nc, nv = d['n_centrales'], d['n_villes']
    x = x.copy()
    for i in range(nc):
        for j in range(nv):
            if rng.random() < taux and x[i][j] > 1e-6:
                k = rng.integers(nc)
                if k != i:
                    delta = rng.uniform(0, x[i][j] * 0.3)
                    x[i][j] -= delta
                    x[k][j] += delta
    return _reparer(x, d)

def algorithme_genetique(d: dict,
                          pop_size: int = 60,
                          n_gen: int = 200,
                          taux_mutation: float = 0.15,
                          taille_tournoi: int = 5,
                          n_elite: int = 3,
                          seed: int = 42) -> dict:
    """
    Paramètres :
        pop_size: taille de la population
        n_gen: nombre de générations
        taux_mutation: probabilité de mutation par gène
        taille_tournoi: nombre d'individus dans chaque tournoi
        n_elite: nombre de meilleurs individus conservés par élitisme
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()

    pop = _init_population(pop_size, d, rng)
    couts = np.array([cout_total(x, d) for x in pop])

    meilleur_idx = np.argmin(couts)
    meilleure_x = pop[meilleur_idx].copy()
    meilleur_cout = couts[meilleur_idx]

    for gen in range(n_gen):
        # Élites
        elite_idx = np.argsort(couts)[:n_elite]
        nouvelle_pop = [pop[i].copy() for i in elite_idx]

        while len(nouvelle_pop) < pop_size:
            p1 = _tournoi(pop, couts, taille_tournoi, rng)
            p2 = _tournoi(pop, couts, taille_tournoi, rng)
            enfant = _croiser(p1, p2, d, rng)
            enfant = _muter(enfant, d, rng, taux_mutation)
            nouvelle_pop.append(enfant)

        pop = nouvelle_pop
        couts = np.array([cout_total(x, d) for x in pop])

        gen_best_idx = np.argmin(couts)
        if couts[gen_best_idx] < meilleur_cout:
            meilleure_x = pop[gen_best_idx].copy()
            meilleur_cout = couts[gen_best_idx]

    return {
        'nom': 'Algorithme Génétique',
        'x': meilleure_x,
        'cout': meilleur_cout,
        'temps': time.time() - t0,
        'generations': n_gen,
    }


if __name__ == '__main__':
    d = charger_donnees()
    res = algorithme_genetique(d)
    afficher_solution(res['x'], d, res['nom'])
    print(f"  Générations : {res['generations']}")
    print(f"  Temps       : {res['temps']:.3f} s")
