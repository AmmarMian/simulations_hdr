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
| `learning_hyperspectral_metrics` | ce que la géométrie seule apporte, avant toute correction | Salinas | implantée |
| `learning_hyperspectral_rmt` | le gain se maintient en aval de l'estimation | Indian Pines | optionnelle |

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

## Partitionnement hyperspectral

L'eqm mesure un critère *interne* : la distance entre la moyenne estimée et la
vraie. Or la thèse du chapitre est que le critère a quitté le modèle, donc
l'estimateur doit être jugé sur la tâche. Deux expériences le font, et elles
isolent deux choses différentes.

`learning_hyperspectral_metrics` pose la question antérieure : **avant toute
correction, quelle part du gain tient à la métrique ?** Trois géométries —
euclidienne, log-euclidienne, affine-invariante — sur les mêmes covariances,
dans la même alternance. `learning_hyperspectral_rmt` pose la question du
chapitre : la correction survit-elle en aval, une fois la géométrie fixée ?

Dans les deux cas, la boucle d'alternation, les redémarrages et le choix du
meilleur par inertie sont écrits **une seule fois** et partagés, et toutes les
méthodes partent de la même partition initiale à graine égale. C'est une
différence avec le protocole publié, où les lignes de base passaient par un
autre optimiseur que la méthode corrigée : l'écart mesuré y additionnait
l'effet de l'estimateur et celui de l'optimiseur, sans que rien ne les sépare.
Les lignes de base sont ici sensiblement meilleures, et l'écart en faveur de la
correction plus étroit — mais il porte sur la seule chose qui doit varier.

Une précaution de lecture : l'inertie compare les redémarrages d'**une**
métrique et rien d'autre, les trois mesurant des longueurs dans des géométries
différentes. Le classement des méthodes se lit sur l'exactitude et la mIoU, qui
sont sur la vérité terrain.

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
<div class="exp-name">learning_hyperspectral_metrics</div>

</div>
<div class="exp-desc">Euclidean, log-Euclidean and affine-invariant K-means on a hyperspectral scene — what the geometry alone buys, before any correction, entirely on the device</div>
<div class="exp-tags"><span class="exp-tag">learning</span><span class="exp-tag">clustering</span><span class="exp-tag">hyperspectral</span><span class="exp-tag">gpu</span></div>
<div class="exp-run"><code>uv run python 3-learning/hyperspectral-metrics/main.py</code></div>
<a class="exp-details-link" href="../../experiments/learning_hyperspectral_metrics/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">learning_hyperspectral_rmt</div>

</div>
<div class="exp-desc">Riemannian K-means segmentation of a hyperspectral scene — SCM, Ledoit-Wolf, non-linear shrinkage and the RMT correction, judged on the ground truth</div>
<div class="exp-tags"><span class="exp-tag">learning</span><span class="exp-tag">random-matrix-theory</span><span class="exp-tag">clustering</span><span class="exp-tag">hyperspectral</span></div>
<div class="exp-run"><code>uv run python 3-learning/hyperspectral-rmt/main.py</code></div>
<a class="exp-details-link" href="../../experiments/learning_hyperspectral_rmt/">Parameters &amp; details →</a>
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
