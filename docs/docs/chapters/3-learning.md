# Chapitre 3 · Apprendre sous un modèle corrigé

Code des figures du chapitre `ch:learning`, celui qui tient entre la détection
et l'apprentissage profond. Sa thèse, en une phrase : faire de l'apprentissage
sur des covariances ne dispense pas du modèle, celui-ci revient sous forme de
**contrainte** (vraisemblance elliptique pénalisée, modèle factoriel de rang
faible) ou de **correction** (théorie des matrices aléatoires appliquée à la
moyenne de Fréchet).

Comme les chapitres 1 et 2, et contrairement au chapitre 4, les expériences
passent par [`hdrlib.core.backend`](../api/core/backend.md) : le même script
tourne sur numpy, torch, cupy ou jax.

## Périmètre

Les deux travaux du chapitre sont des articles de conférence reproduits en fin
de chapitre dans le mémoire, et l'essentiel de leurs campagnes numériques n'est
**pas** rejoué ici — les graphes appris (`animals`, GNSS) sont dans l'article,
avec leur contexte. Ce qui est rejoué est ce qui porte l'argument du mémoire et
que le mémoire doit pouvoir montrer sous sa propre chaîne de provenance.

| Expérience | Ce qu'elle montre | Données | État |
|---|---|---|---|
| `learning_marchenko_pastur` | ce que le régime dimensionnel fait à un spectre : le biais est déterministe, donc corrigible | simulées | faite |
| `learning_frechet_mse` | l'eqm de la moyenne de Fréchet contre $N$ et contre $K$ — et le fait que le gain *croît* avec $K$ | simulées | faite |
| partitionnement hyperspectral | le gain se maintient en aval de l'estimation | Indian Pines | optionnelle |

La correction et la moyenne corrigée sont dans
[`hdrlib.core.rmt`](../api/core/backend.md) ; elles reprennent
[`AmmarMian/icml-rmt-2024`](https://github.com/AmmarMian/icml-rmt-2024). Elles
demandent du float64 — `torch-mps` est donc hors jeu, Metal n'ayant pas de
double précision.

## Marchenko-Pastur

La vraie covariance est l'identité : **toutes** ses valeurs propres valent 1.
Celles de son estimateur s'étalent sur $[(1-\sqrt{c})^2, (1+\sqrt{c})^2]$, et
cet étalement ne dépend que de $c = d/N$ — ajouter des données à $c$ constant
n'est pas la même chose qu'en ajouter à $d$ fixé. C'est ce qui sépare la
correction de la régularisation par rétrécissement : ici le biais est
parfaitement décrit, donc inversible.

<div class="plotly-wrap" data-src="../../assets/data/learning_marchenko_pastur.json" data-title="learning_marchenko_pastur"></div>

Les trois panneaux ne diffèrent que par le nombre d'observations. À $c = 1$ la
densité diverge à l'origine en $1/\sqrt{\lambda}$ ; le cadre suit l'histogramme
et non la courbe, sans quoi le panneau s'écraserait sur sa ligne de base.

## L'eqm de la moyenne

Les deux panneaux répondent à deux questions différentes. Contre le nombre
d'échantillons, l'écart se referme quand $N$ croît : c'est la signature d'un
biais de régime et non d'une variance. Contre le nombre de matrices, il
s'*élargit* — moyenner davantage de matrices réduit la variance mais pas le
biais, qui est commun à toutes les scm ; passé un certain $K$, le biais est
tout ce qui reste et lui seul distingue les méthodes.

Noter aussi *où* chaque méthode corrige : les rétrécissements régularisent
chaque covariance **avant** de moyenner, la méthode rmt corrige la **distance**
que la moyenne minimise. Ce n'est pas le même geste.

<div class="plotly-wrap" data-src="../../assets/data/learning_frechet_mse.json" data-title="learning_frechet_mse"></div>

## Expériences

<!-- experiments-start -->
<div class="exp-chapter">
<div class="exp-group">
<h3 class="exp-group-heading">Learning · Random matrix theory</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">learning_frechet_mse</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">MSE of the Fréchet mean of a set of covariances, against the number of samples and against the number of matrices — SCM, Ledoit-Wolf, OAS, non-linear shrinkage and the RMT correction</div>
<div class="exp-tags"><span class="exp-tag">learning</span><span class="exp-tag">random-matrix-theory</span><span class="exp-tag">frechet-mean</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 3-learning/frechet_mse/main.py</code></div>
<a class="exp-details-link" href="../../experiments/learning_frechet_mse/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">learning_marchenko_pastur</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Marchenko-Pastur law — histogram of the SCM eigenvalues against the theoretical density, for three concentration ratios</div>
<div class="exp-tags"><span class="exp-tag">learning</span><span class="exp-tag">random-matrix-theory</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 3-learning/marchenko_pastur/main.py</code></div>
<a class="exp-details-link" href="../../experiments/learning_marchenko_pastur/">Parameters &amp; details →</a>
</div>
</div>
</div>
</div>
<!-- experiments-end -->
