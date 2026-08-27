# La moyenne accordée au modèle gagne-t-elle, et de combien ?

Sert `prop:spdnet-moyennes-frechet` et le bloc
`% TODO texte --- LE POINT CENTRAL` de `sec:spdnet-batchnorm-moyennes`.
Aucune donnée réelle.

## Pourquoi

La proposition identifie chaque moyenne à la moyenne de Fréchet d'une géométrie
ou d'une divergence : arithmétique ↔ KL à gauche (Wishart), harmonique ↔ KL à
droite (Wishart inverse), GAH ↔ KL symétrisée. Le chapitre veut en faire **un
énoncé de modélisation**, pas une curiosité numérique. Or le seul appui prévu
aujourd'hui est les tableaux de F1 sur trois jeux réels : un argument
*indirect*, sur des données dont on ne connaît pas la loi.

La grille `wishart-inverse` de `eusipco_2026` teste l'énoncé **frontalement** :
on tire d'un Wishart, puis d'un Wishart inverse, et on regarde quelle moyenne
gagne. Elle se relance telle quelle (voir `../NOTE-reproduction.md` §3) :

```sh
python -m eusipco_2026.simulation --wishart-inverse --cpu --n-jobs 8
```

**Ce répertoire n'ajoute qu'un axe** que cette grille ne balaie pas : les degrés
de liberté. Quand $df$ grandit, la loi se concentre autour de sa matrice
d'échelle et le choix de moyenne doit compter de moins en moins. Le mécanisme
apparaît alors comme un **gradient** et non comme deux points — c'est ce qui en
fait un énoncé sur le modèle, et non sur un réglage particulier.

Tout le reste est réutilisé : `run_single_experiment` de `eusipco_2026` fait la
génération, l'entraînement et l'évaluation. Ce fichier ne fait que balayer et
tracer.

## Le résultat de la grille d'origine ($df = 64$, matrices $64\times64$)

| moyenne | données Wishart | données Wishart inverse |
|---|---|---|
| arithmétique (KL gauche) | **90,6 % ± 4,2** | 55,3 % ± 22,1 |
| harmonique (KL droite) | 46,1 % ± 8,5 | **88,9 % ± 5,7** |
| \textsc{gah} (KL symétrisée) | 76,7 % ± 5,6 | 81,4 % ± 5,9 |
| \textsc{armagnac} (GAH adaptative) | 86,7 % ± 2,3 | 87,2 % ± 6,2 |
| géométrique (affine invariante) | 90,0 % ± 4,9 | 88,1 % ± 5,3 |

La moyenne accordée au modèle gagne, l'opposée s'effondre — et l'écart-type de
22 % sur l'arithmétique en Wishart inverse dit la même chose autrement : sous
mauvais modèle, l'apprentissage n'est même plus reproductible d'une graine à
l'autre.

## Deux réserves d'honnêteté, à respecter dans le texte

1. Sur ces données, **la moyenne géométrique est la meilleure des cinq à peu
   près partout** (90,0 / 88,1). Cette figure ne dit donc *pas* « GAH bat la
   géométrique » — elle dit « la moyenne accordée au modèle gagne, les moyennes
   symétriques sont robustes au modèle ». C'est un énoncé plus fort et plus
   défendable. L'argument « la géométrique n'est jamais la meilleure » reste ce
   qu'il est : un fait **des trois jeux réels**, à garder dans le volet 2 et à
   ne surtout pas mélanger avec celle-ci. Bien séparées, les deux se
   renforcent : la simulation établit le mécanisme, le réel montre que sur ces
   données-là l'hypothèse gaussienne n'est pas la bonne.
2. Les matrices d'échelle des classes diffèrent par une perturbation de 25 % :
   la séparation entre classes est un **réglage**, pas une propriété. Le dire.

## Lancer

`eusipco_2026` et `spdnet-datasets` ne sont pas des dépendances de ce dépôt :

```sh
uv pip install git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets.git
uv pip install --no-deps git+https://github.com/Yet-Another-Research-Organisation/eusipco_2026.git
export PYTHONPATH=$HOME/Research/HDR/yetanotherspdnet/src   # cf. ../NOTE-reproduction.md §7
python df_sweep.py
```

`--df 64 96 160 320 640` par défaut ; $df$ doit dépasser `matrix_size - 1`.
