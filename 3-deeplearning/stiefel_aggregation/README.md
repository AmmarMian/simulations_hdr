# À quel ordre projavg et rlavg coïncident-elles ?

Sert `prop:spdnet-federe-equivalence` et `sec:spdnet-federe-agregation` du
chapitre `ch:spdnet`. Aucune donnée, aucun entraînement : 2 secondes.

## Pourquoi

La proposition du chapitre affirme que les deux agrégations coïncident à
$O(\varepsilon^2)$ près quand les poids locaux restent à $O(\varepsilon)$ de
l'itérée globale. Le texte prévoit de l'appuyer sur les courbes de validation
EEG, « qui montrent des trajectoires superposées ». Des trajectoires superposées
établissent que les deux schémas s'accordent ; elles ne mesurent pas **à quel
ordre**, et c'est l'ordre qui décide si la recommandation du chapitre
(`projavg`, moins de constantes, pas besoin de garder l'itérée précédente) est
un arbitrage ou un choix sans contrepartie.

## Résultat

Ordre ajusté sur $\log\lVert\mathrm{projavg}-\mathrm{rlavg}\rVert_F$ contre
$\log\varepsilon$, au-dessus du plancher d'arrondi :

| géométrie | $K=2$ | $K=8$ | $K=32$ |
|---|---|---|---|
| $\mathrm{St}(40,20)$ | 2,98 | 2,99 | 2,99 |
| $\mathrm{St}(128,32)$ | 2,99 | 2,99 | 3,00 |
| $\mathrm{St}(64,60)$ | 2,99 | 2,99 | 3,00 |

**L'ordre est trois, pas deux**, et il l'est sur les neuf configurations. La
proposition est vraie et conservatrice.

Le chiffre à retenir pour le texte : à une dispersion de $10^{-2}$ entre
clients, l'écart entre les deux agrégations vaut $\sim 10^{-6}$ fois le
déplacement de l'agrégat lui-même. À l'échelle où le fédéré travaille, les deux
schémas ne sont pas « proches », ils sont indiscernables.

## Conséquence pour le chapitre

Deux options, à trancher :

1. corriger l'énoncé de `prop:spdnet-federe-equivalence` en $O(\varepsilon^3)$ —
   il faut alors refaire les deux lignes de calcul du bloc `% TODO texte`, le
   terme d'ordre deux devant s'annuler ;
2. garder $O(\varepsilon^2)$, qui est correct, et dire dans le texte que la
   borne est atteinte avec marge, mesure à l'appui.

La seconde est la plus sûre tant que l'annulation du terme d'ordre deux n'est
pas établie au tableau. Ce script mesure, il ne démontre pas.

## Lancer

```sh
export PYTHONPATH=$HOME/Research/HDR/yetanotherspdnet/src   # cf. ../NOTE-reproduction.md §7
python main.py --n_repeats 20
```

Options : `--dimensions 40 20 --dimensions 128 32` (répéter le drapeau),
`--n_clients 2 8 32`, `--dispersions`, `--device cuda`.

Les dispersions s'arrêtent à $10^{-4}$ : en deçà, l'écart atteint le plancher
d'arrondi du float64 (~$10^{-14}$) et cesse de porter un exposant. C'est
`--floor` qui écarte ces points de l'ajustement.
