# Portage de l'implémentation ICML 2024, et ce qu'il a coûté

Sert les deux figures d'eqm de `sec:learning-frechet` et la
`prop:learning-distance-corrigee`. Aucune donnée réelle.

Le code de référence est celui de
[`AmmarMian/icml-rmt-2024`](https://github.com/AmmarMian/icml-rmt-2024). Il est
transposé dans [`hdrlib/core/rmt.py`](../../hdrlib/core/rmt.py) pour passer par
la couche `hdrlib.core.backend`, comme le reste des chapitres 1 à 3.

## Ce qui était censé poser problème, et ce qui en posait vraiment

Le soupçon de départ était que le dépôt de référence utilisait JAX et des
primitives LAPACK impossibles à rendre backend-free. La lecture du code dit
autre chose.

**JAX : rien.** Le mot n'apparaît que dans quelques docstrings
(`jnp.ndarray`), reliquats d'une version antérieure. Le code exécuté est du
numpy pur.

**LAPACK : deux appels, tous deux remplaçables.**

| Référence | Ici | Nature de l'écart |
|---|---|---|
| `dpptrf` — Cholesky en stockage compacté | `linalg.cholesky` | aucun : le stockage compacté est une disposition mémoire, pas une autre factorisation. Vérifié bit à bit |
| `dtrtri` — inverse d'un facteur triangulaire | `linalg.inv` | `dtrtri` exploite la triangularité et fait deux fois moins d'opérations ; le résultat ne diffère que par l'arrondi. Vérifié bit à bit sur nos cas |

`dtrtri` n'existe qu'en double précision (préfixe `d`) : **la référence exige
donc float64 sans le dire**. Le portage le dit, via `rmt.require_double`.

**pymanopt : dépendance de façade.** Le paquet n'est importé que pour la classe
de base de `SPD`, et aucun optimiseur de pymanopt n'est utilisé — la descente
et la recherche linéaire sont écrites à la main dans `mean.py`. Rien à porter.

**scipy.linalg.sqrtm** n'apparaît que dans `SPD.transport`, qui ne sert à aucun
des calculs de moyenne. Non porté.

**scikit-learn** fournit les deux estimateurs à rétrécissement linéaire.
Réécrits ici (dix lignes chacun) plutôt qu'importés, sans quoi toute
l'expérience serait clouée à numpy pour deux formules closes. Attention : `OAS`
de scikit-learn n'est pas la formule de l'article de Chen et al. — elle omet des
facteurs `(1 - 2/p)` — et c'est scikit-learn que la référence appelle, donc
scikit-learn que les figures publiées tracent. C'est cette variante qui est
reproduite.

**`analytical_shrinkage_estimator`** utilise `np.linalg.eig` sur une matrice
symétrique. Remplacé par `eigh` : valeurs propres réelles et déjà triées, au
lieu de valeurs complexes qu'il faut retrier.

## Le vrai point délicat

Ce n'est aucune des substitutions ci-dessus, c'est le gradient. Les formules de
`_rmt_cost_grad` contiennent des expressions comme `mat**3 + eye` ou
`mat**2 - 4*diagL**2`, où `mat` est la matrice des différences de valeurs
propres et s'annule sur la diagonale. Ce ne sont pas des régularisations : les
`+ eye` existent pour que l'entrée diagonale évalue la *limite* de l'expression
hors-diagonale au lieu de 0/0. Elles sont reproduites telles quelles, sans
simplification — les réécrire est le moyen le plus rapide de casser le gradient
silencieusement.

## Validation

```sh
git clone https://github.com/AmmarMian/icml-rmt-2024
uv run --group reference python 3-learning/frechet_mse/validate_against_paper.py \
    --reference icml-rmt-2024/code [--backend torch-cpu]
```

Le script importe les deux implémentations et compare couche par couche. Les
résultats sur `numpy` et `torch-cpu` :

| Couche | Écart relatif | Tolérance |
|---|---|---|
| scm | 0 | 1e-12 |
| distance corrigée | 1e-13 | 1e-10 |
| distance simple | 0 | 1e-10 |
| coût corrigé | 0 | 1e-10 |
| **gradient corrigé** | 0 | 1e-8 |
| rétrécissement analytique (`eigh` vs `eig`) | 4e-10 | 1e-8 |
| Ledoit-Wolf linéaire vs scikit-learn | 0 | 1e-10 |
| OAS vs scikit-learn | 0 | 1e-10 |
| moyenne de Fréchet simple | 5e-15 | 1e-6 |
| moyenne de Fréchet corrigée | 9e-6 | 1e-4 |
| **eqm sur 20 tirages** | 1e-6 | 1e-3 |

Le coût et le gradient — la partie qu'on pouvait craindre — sont **exacts au
bit près**. La seule ligne au-dessus de 1e-8 est la moyenne corrigée elle-même,
et elle mérite une explication.

### Pourquoi la moyenne s'écarte de 1e-5 alors que son gradient est exact

Les deux boucles sont identiques et le restent : à la première itération, le
coût, le gradient et le pas de la recherche linéaire coïncident bit à bit. Puis
elles dérivent, d'environ un chiffre toutes les quelques itérations :

```
it  0  |M diff| = 0.000e+00
it  4  |M diff| = 1.146e-10
it  8  |M diff| = 1.324e-09
it 11  |M diff| = 3.983e-09
```

Le mécanisme est la recherche linéaire par rebroussement. Dès que deux coûts
diffèrent sur leurs derniers bits, elle peut faire une division par deux de plus
ou de moins, ce qui déplace l'itérée suivante de bien plus que l'arrondi qui en
est la cause. Après une trentaine d'itérations au voisinage d'un minimum plat,
l'écart se stabilise autour de 1e-5.

C'est une propriété de l'algorithme, pas du portage : deux exécutions de la
référence sur deux versions de BLAS feraient la même chose. Ce qu'il faut donc
vérifier n'est pas l'égalité des matrices mais **l'égalité de ce que les
figures tracent**, c'est-à-dire l'eqm. Elle est la dernière ligne du tableau, et
elle concorde à 1e-6.

## Limites

`torch-mps` ne peut pas exécuter ce code : Metal n'a pas de float64, et le
gradient perd tous ses chiffres significatifs en simple précision — sans rien
signaler. `rmt.require_double` refuse donc de démarrer plutôt que de rendre une
matrice fausse. Sur Apple Silicon, utiliser `torch-cpu`.
