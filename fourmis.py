
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import charger_donnees, cout_total, afficher_solution
from heuristiques.gloutonne import _solution_initiale_gloutonne


#Construction d'une solution par une fourmi

def _construire_solution(tau: np.ndarray, eta: np.ndarray,
                          d: dict, rng, alpha: float, beta: float) -> np.ndarray:
    nc, nv = d['n_centrales'], d['n_villes']
    x = np.zeros((nc, nv))
    cap_rest = d['capacites_max'].copy()

    ordre_villes = rng.permutation(nv)

    for j in ordre_villes:
        deficit = d['demandes_villes'][j]
        # Utiliser un ordre probabiliste mais sans remise pour garantir la couverture
        centrales_utilisees = set()

        for _ in range(nc):
            if deficit <= 1e-6:
                break
            # Calculer les probs pour les centrales non encore épuisées
            probs = np.zeros(nc)
            for i in range(nc):
                if i in centrales_utilisees:
                    continue
                if cap_rest[i] < 1e-6:
                    continue
                cap_ligne_dispo = d['capacites_lignes'][i][j] - x[i][j]
                if cap_ligne_dispo < 1e-6:
                    continue
                probs[i] = (tau[i][j] ** alpha) * (eta[i][j] ** beta)

            total = probs.sum()
            if total < 1e-12:
                # Plus de capacité probabiliste — prendre ce qui reste sans prob
                for i in range(nc):
                    if deficit <= 1e-6:
                        break
                    if cap_rest[i] < 1e-6:
                        continue
                    cap_ligne_dispo = d['capacites_lignes'][i][j] - x[i][j]
                    if cap_ligne_dispo < 1e-6:
                        continue
                    besoin_prod = deficit / (1 - d['pertes_energie'][i][j])
                    alloc = min(besoin_prod, cap_rest[i], cap_ligne_dispo)
                    if alloc < 1e-9:
                        continue
                    x[i][j] += alloc
                    cap_rest[i] -= alloc
                    deficit -= alloc * (1 - d['pertes_energie'][i][j])
                break

            probs /= total
            i_choisi = int(rng.choice(nc, p=probs))
            centrales_utilisees.add(i_choisi)

            cap_ligne_dispo = d['capacites_lignes'][i_choisi][j] - x[i_choisi][j]
            besoin_prod = deficit / (1 - d['pertes_energie'][i_choisi][j])
            alloc = min(besoin_prod, cap_rest[i_choisi], cap_ligne_dispo)
            alloc = max(alloc, 0.0)

            x[i_choisi][j] += alloc
            cap_rest[i_choisi] -= alloc
            deficit -= alloc * (1 - d['pertes_energie'][i_choisi][j])

    return x


#ACO principal
def aco(d: dict,
        n_fourmis: int = 20,
        n_iter: int = 150,
        alpha: float = 1.0,    # poids phéromone
        beta: float = 3.0,     # poids heuristique
        rho: float = 0.1,      # taux d'évaporation
        q0: float = 0.9,       # poids élite globale dans le dépôt
        seed: int = 42) -> dict:
    """
    Paramètres:
        n_fourmis: nombre de fourmis par itération
        n_iter: nombre d'itérations
        alpha: exposant phéromone (τ^alpha)
        beta: exposant heuristique (η^beta)
        rho: taux d'évaporation [0,1]
        q0: fraction du dépôt attribuée à la meilleure solution globale
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    nc, nv = d['n_centrales'], d['n_villes']
    eta = 1.0 / d['cout_effectif']
    eta = np.clip(eta, 1e-9, None)
    x_init = _solution_initiale_gloutonne(d)
    cout_init = cout_total(x_init, d)
    tau0 = n_fourmis / cout_init
    tau = np.full((nc, nv), tau0)

    meilleure_x = x_init.copy()
    meilleur_cout = cout_init

    for it in range(n_iter):
        solutions = []
        for _ in range(n_fourmis):
            x = _construire_solution(tau, eta, d, rng, alpha, beta)
            c = cout_total(x, d)
            solutions.append((x, c))

        # Meilleure fourmi de l'itération
        x_iter_best, c_iter_best = min(solutions, key=lambda s: s[1])

        if c_iter_best < meilleur_cout:
            meilleure_x = x_iter_best.copy()
            meilleur_cout = c_iter_best
        # Évaporation
        tau *= (1 - rho)
        # Dépôt : on renforce les connexions utilisées (indicateur 0/1)
        indicateur_iter  = (x_iter_best > 1e-6).astype(float)
        indicateur_elite = (meilleure_x > 1e-6).astype(float)
        tau += (1 - q0) * (1.0 / c_iter_best)  * indicateur_iter
        tau += q0        * (1.0 / meilleur_cout) * indicateur_elite
        tau = np.maximum(tau, 1e-10)

    return {
        'nom': 'Colonie de Fourmis (ACO)',
        'x': meilleure_x,
        'cout': meilleur_cout,
        'temps': time.time() - t0,
        'iterations': n_iter,
    }


if __name__ == '__main__':
    d = charger_donnees()
    res = aco(d)
    afficher_solution(res['x'], d, res['nom'])
    print(f"  Itérations : {res['iterations']}")
    print(f"  Temps      : {res['temps']:.3f} s")
