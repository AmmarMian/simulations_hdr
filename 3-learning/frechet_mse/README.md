# Erreur d'estimation de la moyenne de Fréchet

Sert la figure d'eqm de `sec:learning-frechet`. Aucune donnée réelle.

## Ce que ça mesure

La moyenne de Fréchet d'un ensemble de covariances est ce que calcule tout
classifieur au plus proche centroïde et tout $K$-moyennes riemannien. En
pratique on n'a pas les covariances vraies mais leurs estimées, dans le régime
où $d$ et $N$ sont comparables : la moyenne de Fréchet des scm est alors
biaisée.

Cinq estimateurs, deux balayages :

| | |
|---|---|
| `SCM` | moyenne de Fréchet des covariances empiriques |
| `LW`, `OAS` | des covariances rétrécies linéairement |
| `LW-NL` | des covariances rétrécies non linéairement |
| `RMT` | moyenne corrigée par la théorie des matrices aléatoires |

Les quatre premières régularisent chaque covariance **avant** de moyenner ; la
dernière corrige la **distance** que la moyenne minimise. Ce n'est pas le même
geste, et c'est ce que la figure sépare.

## Les deux panneaux

**Contre le nombre d'échantillons $N$** ($K = 10$) : l'écart se referme quand
$N$ croît — 8,1 dB de gain à $N = 65$, 2,2 dB à $N = 300$. Signature d'un biais
de régime, pas d'une variance.

**Contre le nombre de matrices $K$** ($N = 128$) : l'écart s'*ouvre*. La scm
plafonne (12,7 dB à $K = 3$, 8,2 dB à $K = 100$, l'essentiel acquis dès
$K = 20$) pendant que la moyenne corrigée continue de descendre jusqu'à
−4,3 dB. Moyenner réduit la variance, pas le biais : celui-ci est commun à
toutes les scm et survit intact à la moyenne.

C'est le second panneau qui porte l'argument, et c'est celui que l'intuition
rate.

## Lancer

```sh
uv run qanat experiment run learning_frechet_mse --n_features 64 --n-trials 100
```

Une cinquantaine de minutes sur dix cœurs ; le point $K = 100$ domine le coût.
Les tirages sont indépendants et distribués sur un pool (`--n-workers`), et
chacun est graine par `(indice de l'axe, tirage)` : un point peut être rejoué
seul sans replayer le balayage.

## Contraintes

**float64 obligatoire.** Le gradient corrigé divise par des différences de
valeurs propres, et plusieurs termes sont construits pour qu'une entrée
diagonale évalue la limite finie d'une expression qui vaut 0/0 ailleurs. En
simple précision ces termes perdent tous leurs chiffres significatifs sans rien
signaler : la descente rend quand même une matrice, et elle est fausse.
`hdrlib.learning.rmt.require_double` refuse donc de démarrer.

Conséquence pratique : **`torch-mps` ne peut pas exécuter ce code**, Metal
n'ayant pas de float64. Sur Apple Silicon, utiliser `--backend torch-cpu`.

## Convention d'affichage

Les décibels sont des `10*log10(eqm)`, la convention de puissance, qui est la
bonne pour une erreur quadratique.
