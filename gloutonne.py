
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import charger_donnees, cout_total, afficher_solution


def _solution_initiale_gloutonne(d: dict) -> np.ndarray:
    nc, nv = d['n_centrales'], d['n_villes']
    x = np.zeros((nc, nv))

    demandes_rest = d['demandes_villes'].copy()
    cap_rest = d['capacites_max'].copy()
    paires = sorted(
        [(i, j) for i in range(nc) for j in range(nv)],
        key=lambda p: d['cout_effectif'][p[0]][p[1]]
    )

    for i, j in paires:
        if demandes_rest[j] <= 1e-9:
            continue
        if cap_rest[i] <= 1e-9:
            continue
        besoin_prod = demandes_rest[j] / (1 - d['pertes_energie'][i][j])
        cap_dispo = min(cap_rest[i], d['capacites_lignes'][i][j])
        alloc = min(besoin_prod, cap_dispo)

        x[i][j] += alloc
        demandes_rest[j] -= alloc * (1 - d['pertes_energie'][i][j])
        cap_rest[i] -= alloc
    return x


def _reallocation(x: np.ndarray, d: dict) -> np.ndarray:
    nc, nv = d['n_centrales'], d['n_villes']
    improved = True
    while improved:
        improved = False
        for i in range(nc):
            for j in range(nv):
                if x[i][j] < 1e-6:
                    continue
                cout_ij = d['couts_production'][i] + d['cout_transport'][i][j]
                for k in range(nc):
                    if k == i:
                        continue
                    cout_kj = d['couts_production'][k] + d['cout_transport'][k][j]
                    if cout_kj >= cout_ij:
                        continue
                    cap_k = d['capacites_max'][k] - np.sum(x[k])
                    cap_ligne_k = d['capacites_lignes'][k][j] - x[k][j]
                    dispo = min(cap_k, cap_ligne_k)
                    if dispo >= x[i][j] - 1e-9:
                        x[k][j] += x[i][j]
                        x[i][j] = 0.0
                        improved = True
                        break
    return x


def heuristique_gloutonne(d: dict) -> dict:
    t0 = time.time()
    x = _solution_initiale_gloutonne(d)
    x = _reallocation(x, d)
    elapsed = time.time() - t0

    return {
        'nom': 'Heuristique Gloutonne',
        'x': x,
        'cout': cout_total(x, d),
        'temps': elapsed,
    }


if __name__ == '__main__':
    d = charger_donnees()
    res = heuristique_gloutonne(d)
    afficher_solution(res['x'], d, res['nom'])
    print(f"  Temps : {res['temps']:.4f} s")
