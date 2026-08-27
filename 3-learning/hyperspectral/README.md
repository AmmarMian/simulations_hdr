# Partitionnement d'une scène hyperspectrale

Sert la figure de cartes de `sec:learning-frechet`, celle qui juge la
correction **en aval** plutôt que sur elle-même.

## Pourquoi cette expérience et pas seulement l'eqm

La figure d'eqm mesure un critère *interne* : la distance entre la moyenne
estimée et la vraie. Or la thèse du chapitre est que le critère a quitté le
modèle. Il faut donc juger l'estimateur sur la tâche, et une carte de
segmentation le fait voir en un coup d'œil.

## Le pipeline

```
scène → retrait de la moyenne globale → ACP à n_features bandes
      → fenêtre glissante → une covariance par pixel
      → K-moyennes riemannien → comparaison à la vérité terrain
```

C'est là que le régime dimensionnel devient concret : une fenêtre 5×5 sur 5
composantes principales donne **25 échantillons pour 5 variables**, soit
$c = 0{,}2$. Beaucoup de matrices, chacune mal estimée — exactement le régime
du second panneau de la figure d'eqm.

## Ce qui distingue les quatre méthodes

Une seule chose : la distance que minimisent les centroïdes. La boucle
d'alternation, les redémarrages et le choix du meilleur par inertie sont
communs, écrits une fois. Une comparaison mesure donc la correction, et pas
l'optimiseur.

C'est une différence avec le protocole publié, où les lignes de base passaient
par un autre optimiseur que la méthode corrigée. Les lignes de base sont ici
sensiblement meilleures que les publiées, et l'écart en faveur de la correction
plus étroit — mais il porte sur la seule chose qui doit varier.

## Lancer

```sh
uv run qanat experiment run learning_hyperspectral --scene indianpines --n_init 10
uv run qanat experiment run learning_hyperspectral --scene salinas --stride 2 --n_init 5
```

Indian Pines (145×145) prend une petite heure à `stride 1`. Salinas
(512×217) est vingt fois plus grande : `--stride 2` la ramène à un coût
comparable, au prix de la résolution de la carte — la légende doit le dire.

## Données

`download_scene` récupère les `.mat` au premier lancement et vérifie que le
fichier reçu en est bien un : certains miroirs répondent 200 avec une page de
défi, et l'erreur n'apparaîtrait sinon que trois étapes plus loin. Si le
téléchargement échoue, poser les fichiers à la main dans `data/hyperspectral`.

## Contraintes

float64, donc pas de `torch-mps` — voir `hdrlib.core.rmt.require_double`.
Aucun scikit-learn : il est numpy-only, et le pipeline doit pouvoir tourner sur
un backend GPU.
