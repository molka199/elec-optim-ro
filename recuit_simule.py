
import time
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import charger_donnees, cout_total, afficher_solution
from heuristiques.gloutonne import _solution_initiale_gloutonne


def _voisin_aleatoire(x: np.ndarray, d: dict, rng) -> np.ndarray | None:
    nc, nv = d['n_centrales'], d['n_villes']
    actifs = [(i, j) for i in range(nc) for j in range(nv) if x[i][j] > 1e-6]
    if not actifs:
        return None

    for _ in range(50):
        i, j = actifs[rng.integers(len(actifs))]
        k = rng.integers(nc)
        if k == i:
            continue
        ratio = (1 - d['pertes_energie'][i][j]) / (1 - d['pertes_energie'][k][j])

        cap_k = d['capacites_max'][k] - np.sum(x[k])
        cap_ligne_k = d['capacites_lignes'][k][j] - x[k][j]

        max_delta_prime = min(cap_k, cap_ligne_k)
        max_delta = min(x[i][j], max_delta_prime / ratio) if ratio > 1e-9 else x[i][j]

        if max_delta < 1e-6:
            continue

        delta = rng.uniform(max_delta * 0.05, max_delta)
        delta_prime = delta * ratio

        x_new = x.copy()
        x_new[i][j] -= delta
        x_new[k][j] += delta_prime
        return x_new

    return None


def recuit_simule(d: dict,
                  T_init: float = 5000.0,
                  T_min: float = 0.1,
                  alpha: float = 0.995,
                  iter_par_T: int = 30,
                  seed: int = 42) -> dict:
    """
    Paramètres :
        T_init: température initiale
        T_min: température minimale (arrêt)
        alpha: facteur de refroidissement (0.99–0.999)
        iter_par_T: nombre d'itérations par palier de température
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()

    x = _solution_initiale_gloutonne(d)
    meilleure_x = x.copy()
    meilleur_cout = cout_total(x, d)
    cout_courant = meilleur_cout

    T = T_init
    total_iter = 0
    acceptations = 0

    while T > T_min:
        for _ in range(iter_par_T):
            x_new = _voisin_aleatoire(x, d, rng)
            if x_new is None:
                continue
            c_new = cout_total(x_new, d)
            delta = c_new - cout_courant

            if delta < 0 or rng.random() < math.exp(-delta / T):
                x = x_new
                cout_courant = c_new
                acceptations += 1

                if cout_courant < meilleur_cout:
                    meilleure_x = x.copy()
                    meilleur_cout = cout_courant

            total_iter += 1

        T *= alpha

    return {
        'nom': 'Recuit Simulé',
        'x': meilleure_x,
        'cout': meilleur_cout,
        'temps': time.time() - t0,
        'iterations': total_iter,
        'acceptations': acceptations,
        'T_finale': T,
    }


if __name__ == '__main__':
    d = charger_donnees()
    res = recuit_simule(d)
    afficher_solution(res['x'], d, res['nom'])
    print(f"  Itérations   : {res['iterations']}")
    print(f"  Acceptations : {res['acceptations']}")
    print(f"  Temps        : {res['temps']:.3f} s")
