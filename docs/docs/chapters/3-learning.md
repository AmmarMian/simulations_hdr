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
| eqm de la moyenne de Fréchet vs $N$ | l'écart entre la moyenne corrigée et les estimateurs à rétrécissement | simulées | à porter |
| eqm de la moyenne de Fréchet vs $K$ | le gain *croît* avec le nombre de matrices moyennées — moyenner réduit la variance, pas le biais | simulées | à porter |
| partitionnement hyperspectral | le gain se maintient en aval de l'estimation | Indian Pines | optionnelle |

Les trois dernières lignes ont déjà leur code et leurs données côté
`icml-rmt-2024` et côté présentation HCERES ; il reste le portage sous qanat
pour la ligne de provenance et le réhabillage aux couleurs du mémoire.

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

## Expériences

<!-- experiments-start -->
<div class="exp-chapter">
<div class="exp-group">
<h3 class="exp-group-heading">Learning · Random matrix theory</h3>
<div class="exp-grid">
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
