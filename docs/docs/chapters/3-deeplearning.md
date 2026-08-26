# Chapitre 3 · Apprentissage profond sur la variété SPD

Code des figures du chapitre `ch:spdnet`. Contrairement aux chapitres 1 et 2,
celui-ci **ne passe pas par `hdrlib.core.backend`** : les couches sont celles de
[`yetanotherspdnet`](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet),
donc PyTorch pur. Seuls le harnais d'export (`hdrlib.core.exporter`) et le style
de tracé (`hdrlib.core.plot_style`) sont communs.

## Périmètre

Ce que le chapitre rejoue et ce qu'il cite est arrêté dans
[`3-deeplearning/NOTE-reproduction.md`](https://github.com/) — en résumé : rien
n'est rejoué du GPR, des trois jeux réels de la normalisation par lots, ni de
l'EEG fédéré (données non distribuables ou dépôt inexistant), et deux figures
qui ne sont dans aucun article sont produites ici parce qu'elles portent
l'argument du mémoire et coûtent quelques secondes.

## Expériences

| Expérience | Ce qu'elle mesure | Données |
|---|---|---|
| `reeig_spectrum` | ce que le seuil ReEig fait au spectre d'une matrice CovPool, et la borne $1/\varepsilon$ qu'il pose sur la rétropropagation | simulées |
| `reeig_spectrum` (`real_data.py`) | la même mesure sur HDM05 / HyperLeaf / Rices90 | réelles, non distribuables |
| `stiefel_aggregation` | l'ordre auquel les agrégations `projavg` et `rlavg` coïncident | aucune |

Chacune a son `README.md`, qui donne l'énoncé qu'elle sert, le résultat mesuré
et la ligne de commande.

## Matériel

**MPS (Apple Silicon) est refusé explicitement**, et les scripts le disent
plutôt que de se dégrader en silence : MPS n'a pas de `float64`, et
`torch.linalg.eigh` n'y est pas implémenté — c'est l'opération de ReEig, LogEig,
`sqrtm` et des cinq moyennes du chapitre. Avec
`PYTORCH_ENABLE_MPS_FALLBACK=1`, tout retombe sur le CPU une opération à la
fois : `eigh(256×64×64)×10` mesuré à 0,388 s contre 0,364 s en CPU pur.

`--device cpu` ou `--device cuda`.

## Dépendance

`yetanotherspdnet` n'est pas encore une dépendance du dépôt, parce que son
empaquetage amont est cassé (`packages = ["yetanotherspdnet"]` n'embarque aucun
sous-paquet, et l'import échoue sur un faux « circular import »). Le correctif
est prêt sur la branche `fix/whitening-congruence-matrix-grad`. En attendant :

```sh
export PYTHONPATH=$HOME/Research/HDR/yetanotherspdnet/src
```
