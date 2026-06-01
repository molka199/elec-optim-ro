

import math
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import charger_donnees, cout_total, afficher_solution, respecte_contraintes

try:
    from pulp import LpProblem, LpMinimize, LpVariable, LpContinuous, lpSum, LpStatus, value
    PULP_OK = True
except ImportError:
    PULP_OK = False


def resoudre_exact(d: dict) -> dict:
    if not PULP_OK:
        raise ImportError("PuLP non installé. Faites : pip install pulp")

    nc, nv = d['n_centrales'], d['n_villes']
    centrales, villes = d['centrales'], d['villes']

    t0 = time.time()
    prob = LpProblem("elec_distribution", LpMinimize)

    x = LpVariable.dicts("x",
        ((i, j) for i in range(nc) for j in range(nv)),
        lowBound=0, cat=LpContinuous)

    # Fonction objectif
    prob += lpSum(
        (d['couts_production'][i] + d['cout_transport'][i][j]) * x[(i, j)]
        for i in range(nc) for j in range(nv)
    )

    # Contrainte 1 : satisfaire la demande
    for j in range(nv):
        prob += lpSum(
            (1 - d['pertes_energie'][i][j]) * x[(i, j)]
            for i in range(nc)
        ) >= d['demandes_villes'][j]

    # Contrainte 2 : capacité des centrales
    for i in range(nc):
        prob += lpSum(x[(i, j)] for j in range(nv)) <= d['capacites_max'][i]

    # Contrainte 3 : capacité des lignes
    for i in range(nc):
        for j in range(nv):
            prob += x[(i, j)] <= d['capacites_lignes'][i][j]

    prob.solve()
    elapsed = time.time() - t0

    sol = np.array([[value(x[(i, j)]) or 0.0 for j in range(nv)] for i in range(nc)])
    cout = math.ceil(value(prob.objective) * 100) / 100

    return {
        'nom': 'Programmation Linéaire (exacte)',
        'x': sol,
        'cout': cout,
        'statut': LpStatus[prob.status],
        'temps': elapsed,
    }


if __name__ == '__main__':
    d = charger_donnees()
    res = resoudre_exact(d)
    afficher_solution(res['x'], d, res['nom'])
    print(f"  Statut solveur : {res['statut']}")
    print(f"  Temps          : {res['temps']:.3f} s")
