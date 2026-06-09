# HDR Chapitre 2 

Ce répertoire contient le code pour reproduire les résultats présentés dans le chapitre 2 de ma dissertation de HDR.

## Installation

```sh
uv sync                                  # base deps (numpy, torch-cpu, scipy…)
uv sync --extra cupy                     # CuPy / CUDA GPU
uv sync --extra jax                      # JAX CPU
uv sync --extra jax-cuda                 # JAX CUDA GPU
uv sync --extra cupy --extra jax         # combine extras freely
```

## Data

Pour certaines figures, il est nécessaire d'avoir des données réelles en Sonar et SAR:
* Pour les données Sonar, je ne peux malheureusement pas les distribuer.
* Pour les données SAR, se mettre dans `./data/` et exécuter `bash download_sar.sh`

### Préparation des données pour les scripts d'expériences

Les scripts `compute_cd_online.py`, `compute_cd_offline.py` et `compute_cd_kronecker_offline.py`
attendent les données dans un format **temps en premier** `(n_times, n_rows, n_cols, n_features)`
pour un accès mémoire efficace. Après avoir téléchargé les données SAR, convertir chaque fichier
avec le script `prepare_data.py` :

```bash
uv run sar_experiments/prepare_data.py data/SAR/scene1.npy
uv run sar_experiments/prepare_data.py data/SAR/scene2.npy
uv run sar_experiments/prepare_data.py data/SAR/Scene4_cropped.npy
```

Chaque commande crée un fichier `<nom>_time_first.npy` dans le même répertoire. Les scripts
d'expériences vérifient automatiquement la présence de ce fichier et affichent les instructions
si ce n'est pas le cas.


