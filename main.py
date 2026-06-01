"""
Script principal de comparaison – Optimisation du réseau de distribution d'électricité.

Lance tous les algorithmes, affiche un tableau comparatif et sauvegarde
les résultats dans results/comparaison.csv.

Usage :
    python main.py
    python main.py --no-exact     # si PuLP non installé
"""

import argparse
import math
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import charger_donnees, respecte_contraintes

# ── Imports des algorithmes ────────────────────────────────────────────────────
from heuristiques.gloutonne             import heuristique_gloutonne
from metaheuristiques.tabou             import recherche_tabou
from metaheuristiques.recuit_simule     import recuit_simule
from metaheuristiques.fourmis           import aco
from metaheuristiques.genetique         import algorithme_genetique


SEPARATEUR = "─" * 72


def arrondir(v: float) -> float:
    return math.ceil(v * 100) / 100


def lancer_tous(d: dict, avec_exact: bool = True) -> list:
    resultats = []

    algorithmes = [
        ("Heuristique Gloutonne",    lambda: heuristique_gloutonne(d)),
        ("Recherche Tabou",          lambda: recherche_tabou(d, max_iter=500)),
        ("Recuit Simulé",            lambda: recuit_simule(d)),
        ("Colonie de Fourmis (ACO)", lambda: aco(d, n_fourmis=20, n_iter=150)),
        ("Algorithme Génétique",     lambda: algorithme_genetique(d, pop_size=60, n_gen=200)),
    ]

    if avec_exact:
        try:
            from exact.pl_solver import resoudre_exact
            algorithmes.insert(0, ("PL Exacte (PuLP)", lambda: resoudre_exact(d)))
        except ImportError:
            print("⚠  PuLP non disponible — résolution exacte ignorée.")

    for nom, fn in algorithmes:
        print(f"\n▶ {nom} ...", end=' ', flush=True)
        try:
            res = fn()
            verif = respecte_contraintes(res['x'], d)
            res['faisable'] = verif['ok']
            res['ecart_pct'] = None   # rempli après
            resultats.append(res)
            cout_str = f"{arrondir(res['cout']):.2f} €"
            ok_str = "✓" if verif['ok'] else f"✗ ({verif['details']})"
            print(f"  Coût = {cout_str}   Contraintes : {ok_str}   "
                  f"({res['temps']:.3f} s)")
        except Exception as e:
            print(f"  ERREUR : {e}")

    return resultats


def afficher_tableau(resultats: list, cout_optimal: float | None):
    print(f"\n\n{'':=<72}")
    print("  TABLEAU COMPARATIF")
    print(f"{'':=<72}")
    print(f"  {'Algorithme':<30} {'Coût (€)':>12} {'Écart opt.':>10} "
          f"{'Faisable':>9} {'Temps (s)':>10}")
    print(SEPARATEUR)

    for r in resultats:
        cout_r = arrondir(r['cout'])
        if cout_optimal is not None:
            ecart = (cout_r - cout_optimal) / cout_optimal * 100
            ecart_str = f"+{ecart:.2f}%" if ecart > 0 else "  0.00%"
        else:
            ecart_str = "   n/a"
        faisable_str = "  Oui" if r['faisable'] else "  Non"
        print(f"  {r['nom']:<30} {cout_r:>12.2f} {ecart_str:>10} "
              f"{faisable_str:>9} {r['temps']:>10.3f}")

    print(SEPARATEUR)


def sauvegarder_csv(resultats: list, path: str):
    rows = []
    for r in resultats:
        rows.append({
            'Algorithme':  r['nom'],
            'Cout_total':  arrondir(r['cout']),
            'Faisable':    r['faisable'],
            'Temps_s':     round(r['temps'], 4),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, sep=';')
    print(f"\n  Résultats sauvegardés dans : {path}")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comparaison algorithmes — réseau électrique")
    parser.add_argument('--no-exact', action='store_true', help="Ignorer la résolution exacte PuLP")
    args = parser.parse_args()

    print("\n" + "="*72)
    print("  Optimisation d'un réseau de distribution d'électricité")
    print("  6 centrales  ×  8 villes  — minimisation du coût de production + transport")
    print("="*72)

    d = charger_donnees()
    resultats = lancer_tous(d, avec_exact=not args.no_exact)

    if not resultats:
        print("Aucun résultat à afficher.")
        return

    # Coût optimal = PL exacte si disponible, sinon minimum observé
    faisables = [r for r in resultats if r['faisable']]
    cout_optimal = None
    for r in resultats:
        if 'PL' in r['nom'] or 'exacte' in r['nom'].lower():
            cout_optimal = arrondir(r['cout'])
            break
    if cout_optimal is None and faisables:
        cout_optimal = arrondir(min(r['cout'] for r in faisables))

    # Calcul des écarts
    for r in resultats:
        if cout_optimal:
            r['ecart_pct'] = (arrondir(r['cout']) - cout_optimal) / cout_optimal * 100

    afficher_tableau(resultats, cout_optimal)

    csv_path = os.path.join(os.path.dirname(__file__), 'results', 'comparaison.csv')
    sauvegarder_csv(resultats, csv_path)


if __name__ == '__main__':
    main()
