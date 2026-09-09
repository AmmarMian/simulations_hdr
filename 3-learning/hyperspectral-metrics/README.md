# Partitionnement hyperspectral : ce que la géométrie apporte

Reprend le protocole de l'exemple `image-radar` de pyRiemann — même scène, même
fenêtre, même estimateur — sans pyRiemann, et sur le device.

## La question

L'expérience voisine (`3-learning/hyperspectral-rmt`) demande si la correction
RMT survit en aval. Celle-ci pose la question antérieure : **avant toute
correction, quelle part du gain tient à la métrique ?** Même scène, mêmes
fenêtres, mêmes covariances, même alternance — seule change la géométrie dans
laquelle les centroïdes sont des moyennes.

| métrique | centroïde | forme close | coût |
|---|---|---|---|
| `euclid` | moyenne arithmétique des covariances | oui | un produit matriciel |
| `logeuclid` | moyenne des logarithmes matriciels | oui | N décompositions, **une fois** |
| `riemann` | moyenne de Karcher (affine-invariante) | non | N·K décompositions **par itération** |

`logeuclid` n'est pas dans l'exemple pyRiemann. Il est ajouté parce qu'il coûte
presque rien une fois les logarithmes en cache et qu'il sépare deux effets que
`euclid` contre `riemann` confond : respecter la positivité des valeurs propres,
et être invariant par transformation affine.

## Le pipeline

```
scène → retrait de la moyenne globale → normalisation d'échelle
      → ACP à n_features bandes → fenêtre glissante → une covariance par pixel
      → K-moyennes (euclid | logeuclid | riemann) → comparaison à la vérité terrain
```

Une fenêtre 5×5 sur 5 composantes donne **25 échantillons pour 5 variables**,
soit $c = 0{,}2$ : beaucoup de matrices, chacune mal estimée. C'est le régime que
la correction de l'expérience voisine vise ; ici il sert de décor, pas de sujet.

## Tout sur le device

Le cube traverse le bus une fois, les étiquettes reviennent une fois. Entre les
deux, `hdrlib.learning.clustering.spd_kmeans` ne relit que deux scalaires par
itération — la fraction de points qui ont changé de groupe, qui décide de
l'arrêt, et l'inertie, qui choisit le meilleur redémarrage. Ce sont des branches
du programme : elles doivent devenir des nombres Python.

Ce qui rend cela possible, c'est l'absence de correction. Aucune des trois
métriques n'a besoin des échantillons dont la covariance est issue, donc les
covariances sont formées **une fois**, avant les redémarrages, et les fenêtres
sont libérées avant que la boucle ne commence — 108 Mo rendus contre 22 Mo
gardés sur Salinas. L'expérience voisine ne peut pas faire cela : la moyenne
corrigée relit les échantillons.

L'affectation est groupée sur les centroïdes plutôt que bouclée, et la
ré-estimation est un produit contre une matrice d'appartenance one-hot, qui
moyenne les K groupes d'un coup. Le détail sordide : aucune primitive de
`scatter` n'est commune à numpy, torch, cupy et jax — un produit matriciel, si.

## float64, obligatoire

Les trois métriques finissent sur les valeurs propres d'une covariance 5×5
estimée sur 25 échantillons, et deux d'entre elles en prennent le logarithme. En
float32 le signe de la plus petite valeur propre n'est pas fiable, et son
logarithme est soit un grand négatif soit un NaN — silencieusement. La
vérification est faite après le transfert, pas avant : c'est le transfert qui
dégrade.

Conséquences : pas de `torch-mps` (Metal n'a pas le float64, et
`get_data_on_device` rétrograde sans rien dire), et **pas de `jax-*` non plus** —
rien dans ce dépôt n'appelle `jax.config.update("jax_enable_x64", True)`, donc
JAX calculerait tout en simple précision sans même un avertissement. Le backend
les refuse explicitement.

## Lancer

```sh
uv run qanat experiment run learning_hyperspectral_metrics --scene salinas --backend torch-cuda
uv run qanat experiment run learning_hyperspectral_metrics --scene indianpines --backend torch-cuda --n_init 20
```

## Lire les scores

`scores.json` contient les temps **et** la description de la carte. Les deux se
lisent ensemble : le float64 tourne à la moitié du float32 sur une carte de
centre de calcul et au soixante-quatrième sur une carte de station de travail.
`riemann` est limité par les décompositions propres, les deux métriques plates
par les produits matriciels — donc le *classement par le temps* est autant une
propriété de la carte que de la méthode.

L'inertie compare les redémarrages d'**une** métrique et rien d'autre : les trois
mesurent des longueurs dans des géométries différentes. Le classement des
métriques, c'est l'exactitude et la mIoU, qui sont sur la vérité terrain.
