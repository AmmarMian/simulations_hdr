# Ce que coûte la rétropropagation manuelle de la couche de normalisation

Volet 1 de `sec:spdnet-batchnorm-resultats` (`ch:spdnet`). Aucune donnée.

**Ce n'est pas une reproduction mais une réécriture** : aucun des cinq dépôts
SPDnet ne contient de code de mesure de temps ou de mémoire. Les trois figures
coût/mémoire de `PREP26b` ont été produites par du code jamais versionné.

## La mesure de mémoire

L'article mesure `torch.cuda.max_memory_allocated`, qui exige un GPU et mêle la
quantité d'intérêt au comportement de l'allocateur. Ce que
`prop:spdnet-grad-geo` prédit est plus étroit et exactement mesurable : la
différentiation automatique doit **retenir les $\Niterm$ itérées** du point fixe
`eq:spdnet-geometrique-iteration`, là où la formule manuelle recalcule.

`torch.autograd.graph.saved_tensors_hooks` intercepte chaque tenseur que le
graphe garde en vie ; en sommer les tailles mesure précisément cela, sur
n'importe quel matériel, et c'est directement attribuable. Le pic de
l'allocateur est relevé en plus sous CUDA, pour comparaison avec l'article.

## Le mécanisme, isolé (`--sweep iterations`, hors article)

Le balayage qui explique les trois autres. Matrices 64, batch 64, moyenne
géométrique :

| $\Niterm$ | manuel | autograd | rapport |
|---|---|---|---|
| 1 | 5,47 Mio | 7,49 Mio | 1,37× |
| 5 | 7,05 Mio | 18,64 Mio | 2,64× |
| 20 | 20,72 Mio | 60,38 Mio | **2,91×** |

Le facteur 2 annoncé dans le chapitre est le régime à peu d'itérations ; il
croît avec $\Niterm$, ce qui **est** l'énoncé de `prop:spdnet-grad-geo` — deux
équations de Sylvester *par itération* pour la moyenne géométrique, contre deux
en tout pour GAH.

## Les trois figures de l'article

`--sweep size` (batch 64), `--sweep batch` (matrices 64), `--sweep depth`
(matrices 64, batch 64). Extraits, CPU float64, conditionnement $10^5$ :

| balayage | point | moyenne | mém. manuel | mém. autograd | rapport |
|---|---|---|---|---|---|
| taille | 512 | géométrique | 461,5 Mio | 1115,7 Mio | 2,42× |
| taille | 512 | GAH | 312,0 Mio | 404,7 Mio | 1,30× |
| batch | 512 | géométrique | 54,4 Mio | 134,9 Mio | 2,48× |
| profondeur | 32 | géométrique | 280,5 Mio | 873,7 Mio | **3,12×** |
| profondeur | 32 | GAH | 70,8 Mio | 144,1 Mio | 2,04× |

Le rapport **croît avec les trois paramètres**, comme l'annonce le chapitre, et
il est systématiquement plus élevé pour la moyenne géométrique que pour GAH —
c'est le point fixe déroulé qui le porte.

## Le temps : à ne pas survendre

Le bloc `% TODO` du chapitre demande de « le dire franchement ». Mesuré :

| balayage | point | moyenne | t manuel | t autograd |
|---|---|---|---|---|
| taille | 512 | géométrique | 6001 ms | 6153 ms |
| taille | 512 | GAH | 753 ms | 909 ms |
| profondeur | 32 | géométrique | 2684 ms | 2767 ms |

Les temps sont **comparables**, à quelques pour cent près, et l'écart va parfois
dans un sens parfois dans l'autre. **L'argument est la mémoire et la robustesse
numérique, pas la vitesse.** Le gain de temps que rapporte l'article sur données
réelles (temps divisé par 2 à 5) vient du choix de moyenne — GAH est un ordre de
grandeur plus rapide que la géométrique, ici comme là-bas — et non de la
dérivation manuelle.

## Lancer

```sh
export PYTHONPATH=$HOME/Research/HDR/yetanotherspdnet/src   # cf. ../NOTE-reproduction.md §7
python main.py --sweep size   --n_repeats 5
python main.py --sweep batch  --n_repeats 5
python main.py --sweep depth  --n_repeats 5
python main.py --sweep iterations --n_repeats 5
```

`--device cuda` ajoute le pic d'allocateur à côté de la mémoire retenue. MPS est
refusé (pas de float64, pas de `linalg.eigh`).

Les chiffres ci-dessus sont à `--n_repeats 3` sur CPU ; les mémoires sont
déterministes, les temps sont à relancer sur la machine de mesure.
