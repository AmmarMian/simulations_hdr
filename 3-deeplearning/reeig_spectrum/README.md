# L'effet de ReEig sur le spectre

Sert `rem:spdnet-covpool-regime` et `rem:spdnet-reeig-retrecissement` du
chapitre `ch:spdnet`. Remplace la figure GPR abandonnée (§5 de
`../NOTE-reproduction.md`) : elle ne prétend rien sur la performance en
classification, elle mesure ce que ReEig fait au spectre et ce que ça achète à
la rétropropagation.

## L'énoncé

Rétropropager à travers une couche spectrale multiplie l'erreur entrante par la
matrice de Loewner de `prop:spdnet-diffm`,

$$\mathbf{G}_{ij} = \frac{h(\lambda_i)-h(\lambda_j)}{\lambda_i-\lambda_j}.$$

Pour $h=\log$, le théorème des accroissements finis donne
$|\mathbf{G}_{ij}| \le 1/\min(\lambda_i,\lambda_j)$, donc

$$\max_{ij}|\mathbf{G}_{ij}| = 1/\lambda_{\min}.$$

L'instabilité de `prop:spdnet-gradevd` est donc gouvernée par la **plus petite**
valeur propre, et par rien d'autre. Une couche ReEig de seuil $\varepsilon$
impose $\lambda_{\min}\ge\varepsilon$, donc **borne le facteur par
$1/\varepsilon$**. C'est le sens précis dans lequel ReEig est un rétrécissement
spectral : elle paie un biais sur les petites valeurs propres pour acheter une
borne sur le gradient.

## Ce qui est mesuré

| | script | données |
|---|---|---|
| mécanisme, balayage $(N_{pix}/N_{filtre}, \text{décroissance})$ | `main.py` | simulées |
| ce que ça donne sur les jeux du chapitre, balayage $\varepsilon$ | `real_data.py` | HDM05 / HyperLeaf / Rices90 |

`main.py` modélise `eq:spdnet-covpool` pour ce qu'elle est — une covariance
empirique sur $N_{pix}$ positions pour $N_{filtre}$ canaux — et balaie deux
axes, parce que les deux comptent indépendamment : le ratio d'échantillonnage
dit de combien le spectre empirique tombe sous le vrai, la décroissance dit où
le vrai était déjà.

## Résultat (simulé, `--n_trials 100`, $N_{filtre}=256$, $\varepsilon=10^{-4}$)

| décroissance | ratio | $\lambda_{\min}$ | cond. | % écrêtées | Loewner | + ReEig |
|---|---|---|---|---|---|---|
| $10^2$ | 3 | 4,7e-03 | 2,7e+02 | 0,0 % | 2,1e+02 | 2,1e+02 |
| $10^6$ | 0,75 | −3,1e-18 | **singulière** | 43,0 % | **non défini** | 1,0e+04 |
| $10^6$ | 3 | 5,9e-07 | 1,8e+06 | 35,2 % | 1,7e+06 | **1,0e+04** |

Trois choses à retenir :

1. **Sous le ratio 1, la matrice est singulière** — le centrage de
   `eq:spdnet-covpool` plafonne son rang à $N_{pix}-1$ — et LogEig n'est pas
   défini du tout, quelle que soit la décroissance. Ce n'est pas un mauvais
   conditionnement, c'est une absence de valeur.
2. **Au point de fonctionnement de l'article** (ratio $\approx 3$) et au
   conditionnement des données du chapitre ($\sim 10^6$ ; HDM05 est annoncé à
   $9{,}1\times10^5$), ReEig écrête **un tiers du spectre** et divise le
   facteur de Loewner par **170**.
3. **À décroissance faible, ReEig ne fait rien** (0 % d'écrêtage). L'affirmation
   du chapitre selon laquelle ReEig est une nécessité et non un raffinement est
   donc vraie *du fait de la décroissance spectrale des données*, pas du régime
   dimensionnel seul. À dire dans ces termes.

## Lancer

```sh
uv sync                     # depuis la racine de simulations_hdr
```

`yetanotherspdnet` n'est pas encore une dépendance du dépôt et son installation
amont est cassée (cf. §7 de `../NOTE-reproduction.md`). En attendant que le
correctif soit fusionné :

```sh
export PYTHONPATH=$HOME/Research/HDR/yetanotherspdnet/src
# ou, une fois la branche fusionnee :
#   uv pip install git+https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet.git
```

Puis, depuis ce répertoire :

```sh
python main.py --n_trials 100                     # figure simulée
python main.py --device cuda --n_trials 1000      # sur GPU
```

**MPS est refusé explicitement** par `common.py` : pas de float64, et
`linalg.eigh` n'y est pas implémenté, donc chaque couche spectrale retomberait
sur le CPU une opération à la fois. Mesuré : 0,388 s contre 0,364 s en CPU pur.
CPU ou CUDA.

### Données réelles

`real_data.py` n'a **pas pu être exécuté** ici (les jeux ne sont pas sur cette
machine). Vérifier d'abord qu'il tourne, sans aucune donnée :

```sh
python real_data.py --self-test
```

puis, sur une machine qui les a — il faut aussi `spdnet-datasets`, que le dépôt
n'installe pas :

```sh
uv pip install git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets.git
python real_data.py --dataset hdm05 --data-root "$DATA_ROOT/HDM05" \
                    --scaling-factor 190.0 --device cuda
```

Le script recoupe le conditionnement qu'il mesure avec celui annoncé dans le
chapitre et signale un écart, sur la dimension comme sur le conditionnement.

**Attention au facteur d'échelle.** $\varepsilon$ est un seuil *absolu*, donc
non invariant d'échelle : les configs de `sigpro_2026` appliquent un
`scaling_factor` (190,0 pour HDM05), et multiplier les matrices par une
constante multiplie le spectre sans déplacer $\varepsilon$. Un pourcentage de
valeurs propres écrêtées ne se lit qu'en regard de l'échelle du spectre, que le
script affiche à côté.
