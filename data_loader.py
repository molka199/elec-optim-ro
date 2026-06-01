"""
Chargement et validation des données du problème d'optimisation
réseau de distribution d'électricité (6 centrales → 8 villes).
"""

import numpy as np
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', r'C:\Users\molka\Downloads\Data Sujet 01.xlsx')

N_CENTRALES = 6
N_VILLES = 8


def charger_donnees(path: str = DATA_PATH) -> dict:
    """Charge toutes les données depuis le fichier Excel et retourne un dict."""
    data = pd.read_excel(path, sheet_name='Feuil1', header=None)

    capacites_max     = data.iloc[3:9,   1].values.astype(float)
    couts_production  = data.iloc[3:9,   2].values.astype(float)
    demandes_villes   = data.iloc[12:20, 1].values.astype(float)
    cout_transport    = data.iloc[23:29, 1:9].values.astype(float)
    pertes_energie    = data.iloc[32:38, 1:9].values.astype(float) / 100
    capacites_lignes  = data.iloc[41:47, 1:9].values.astype(float)

    centrales = [f'C{i+1}' for i in range(N_CENTRALES)]
    villes    = [f'V{j+1}' for j in range(N_VILLES)]

    # Coût effectif par MW livré = (prod + transport) / (1 - perte)
    cout_effectif = (couts_production.reshape(-1, 1) + cout_transport) / (1 - pertes_energie)

    return {
        'capacites_max':    capacites_max,
        'couts_production': couts_production,
        'demandes_villes':  demandes_villes,
        'cout_transport':   cout_transport,
        'pertes_energie':   pertes_energie,
        'capacites_lignes': capacites_lignes,
        'cout_effectif':    cout_effectif,
        'centrales':        centrales,
        'villes':           villes,
        'n_centrales':      N_CENTRALES,
        'n_villes':         N_VILLES,
    }


def cout_total(x: np.ndarray, d: dict) -> float:
    """Calcule le coût total d'une matrice d'allocation x (6×8)."""
    return float(np.sum((d['couts_production'].reshape(-1, 1) + d['cout_transport']) * x))


def respecte_contraintes(x: np.ndarray, d: dict, tol: float = 1e-6) -> dict:
    """
    Vérifie les 3 groupes de contraintes.
    Retourne un dict {'ok': bool, 'details': str}.
    """
    msgs = []

    # C1 : demande satisfaite après pertes
    livraison = np.sum((1 - d['pertes_energie']) * x, axis=0)
    deficit = d['demandes_villes'] - livraison
    if np.any(deficit > tol):
        idx = np.where(deficit > tol)[0]
        msgs.append(f"Demande non satisfaite pour {[d['villes'][j] for j in idx]}")

    # C2 : capacité centrales
    prod_total = np.sum(x, axis=1)
    depass = prod_total - d['capacites_max']
    if np.any(depass > tol):
        idx = np.where(depass > tol)[0]
        msgs.append(f"Capacité dépassée pour {[d['centrales'][i] for i in idx]}")

    # C3 : capacité lignes
    depass_ligne = x - d['capacites_lignes']
    if np.any(depass_ligne > tol):
        msgs.append("Capacité ligne dépassée")

    # C4 : non-négativité
    if np.any(x < -tol):
        msgs.append("Valeurs négatives détectées")

    ok = len(msgs) == 0
    return {'ok': ok, 'details': '; '.join(msgs) if msgs else 'Toutes contraintes respectées'}


def afficher_solution(x: np.ndarray, d: dict, nom: str = "Solution"):
    """Affiche proprement une matrice d'allocation."""
    import math
    cout = cout_total(x, d)
    verif = respecte_contraintes(x, d)
    print(f"\n{'='*55}")
    print(f"  {nom}")
    print(f"  Coût total : {math.ceil(cout * 100) / 100:.2f} €")
    print(f"  Contraintes : {verif['details']}")
    print(f"{'='*55}")
    header = "        " + "".join(f"{v:>8}" for v in d['villes'])
    print(header)
    for i in range(d['n_centrales']):
        row = f"{d['centrales'][i]:>6}  " + "".join(
            f"{x[i][j]:>8.2f}" if x[i][j] > 1e-6 else "       -"
            for j in range(d['n_villes'])
        )
        print(row)
