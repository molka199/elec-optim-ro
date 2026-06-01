

import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import charger_donnees, cout_total, afficher_solution
from heuristiques.gloutonne import _solution_initiale_gloutonne


# ── Voisinage ──────────────────────────────────────────────────────────────────

def _generer_voisins(x: np.ndarray, d: dict, n_voisins: int = 60):
    """
    Génère des voisins par transfert partiel avec compensation des pertes.
    Mouvement : déplace δ MW de prod du trajet (i→j) vers (k→j),
    en ajustant δ' = δ*(1-p_ij)/(1-p_kj) pour conserver la livraison nette.
    """
    nc, nv = d['n_centrales'], d['n_villes']
    voisins = []
    rng = np.random.default_rng()

    candidats = [(i, j) for i in range(nc) for j in range(nv) if x[i][j] > 1e-6]
    if not candidats:
        return voisins

    for _ in range(n_voisins * 4):
        if len(voisins) >= n_voisins:
            break
        i, j = candidats[rng.integers(len(candidats))]
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

        delta = rng.uniform(max_delta * 0.1, max_delta)
        delta_prime = delta * ratio
        x_new = x.copy()
        x_new[i][j] -= delta
        x_new[k][j] += delta_prime

        c = cout_total(x_new, d)
        voisins.append((x_new, (i, j, k), c))

    return voisins


# ── Algorithme principal ───────────────────────────────────────────────────────

def recherche_tabou(d: dict,
                    max_iter: int = 500,
                    taille_tabou: int = 20,
                    n_voisins: int = 80,
                    stagnation_max: int = 80,
                    seed: int = 42) -> dict:
    np.random.seed(seed)
    t0 = time.time()

    x = _solution_initiale_gloutonne(d)
    meilleure_x = x.copy()
    meilleur_cout = cout_total(x, d)
    cout_courant = meilleur_cout

    liste_tabou = []       # liste de mouvements (i,j,k) interdits
    stagnation = 0

    for it in range(max_iter):
        voisins = _generer_voisins(x, d, n_voisins)
        if not voisins:
            break

        # Trier par coût croissant
        voisins.sort(key=lambda v: v[2])

        meilleur_voisin, meilleur_mvt, cout_v = None, None, float('inf')

        for x_v, mvt, c_v in voisins:
            is_tabou = mvt in liste_tabou
            # Critère d'aspiration
            if is_tabou and c_v >= meilleur_cout:
                continue
            if c_v < cout_v:
                meilleur_voisin, meilleur_mvt, cout_v = x_v, mvt, c_v

        if meilleur_voisin is None:
            break

        x = meilleur_voisin
        cout_courant = cout_v

        # Mise à jour liste tabou
        liste_tabou.append(meilleur_mvt)
        if len(liste_tabou) > taille_tabou:
            liste_tabou.pop(0)

        # Mise à jour meilleure solution
        if cout_courant < meilleur_cout:
            meilleure_x = x.copy()
            meilleur_cout = cout_courant
            stagnation = 0
        else:
            stagnation += 1

        # Diversification : redémarre depuis la meilleure solution
        if stagnation >= stagnation_max:
            x = meilleure_x.copy()
            stagnation = 0

    return {
        'nom': 'Recherche Tabou',
        'x': meilleure_x,
        'cout': meilleur_cout,
        'temps': time.time() - t0,
        'iterations': it + 1,
    }


if __name__ == '__main__':
    d = charger_donnees()
    res = recherche_tabou(d, max_iter=500)
    afficher_solution(res['x'], d, res['nom'])
    print(f"  Itérations : {res['iterations']}")
    print(f"  Temps      : {res['temps']:.3f} s")
