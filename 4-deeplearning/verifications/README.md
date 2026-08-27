# Vérifications de session (2026-08-26)

Scripts jetables qui étayent `../NOTE-reproduction.md`. Ce ne sont **pas** des
expériences qanat : pas d'export, pas de `make_mc_parser`. À réécrire au format
du dépôt si les figures correspondantes sont retenues.

Prérequis (cf. §7 de la note — l'installation du paquet est cassée) :

```sh
uv venv .venv --python 3.11
VIRTUAL_ENV=$PWD/.venv uv pip install torch scipy
git clone https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet
export PYTHONPATH=$PWD/yetanotherspdnet/src
```

| Script | Ce qu'il montre | Note |
|---|---|---|
| `smoke.py` | les deux chemins de gradient (`use_autograd`) tournent pour les 5 moyennes ; temps par passage | §6 |
| `iso.py` | le désaccord manuel/autograd est localisé dans `Whitening` / `CongruenceSPD`, sur le gradient de la **matrice** seulement | §8 |
| `equiv.py` | `||projavg - rlavg||_F` en fonction de ε sur `St(40,20)` : pente 3 | §4 |

La grille Wishart / Wishart inverse (§3) n'a pas de script ici : elle se relance
telle quelle depuis `eusipco_2026`, après avoir repointé la dépendance
`dev_yetanotherspdnet` vers le dépôt public.

```sh
python -m eusipco_2026.simulation --wishart-inverse --cpu --n-jobs 8 --output-dir results/wi
```
