# Plan de travail — chapitre `ch:spdnet`

Établi le 2026-08-26 à partir de `NOTE-reproduction.md`. Exécution en autonomie.
Consigne : réutiliser au maximum les dépôts existants, ne réécrire que ce qui
manque.

État : `[ ]` à faire · `[~]` en cours · `[x]` fait · `[!]` bloqué / abandonné

---

## Phase 0 — Reconnaissance  `[x]`

- `[x]` **Verdict MPS : inutilisable.** `float64` absent de MPS (la bibliothèque
  est en float64 partout) ; `torch.linalg.eigh` **non implémenté** sur MPS —
  c'est l'opération centrale de ReEig, LogEig, sqrtm et des cinq moyennes ;
  `linalg.svd` retombe aussi sur CPU (donc la rétraction polaire aussi). Avec
  `PYTORCH_ENABLE_MPS_FALLBACK=1` : 0,388 s contre 0,364 s CPU sur
  `eigh(256×64×64)×10`. **Tout le chapitre reste sur CPU float64.**
- `[x]` Inventaire données : pas de HDM05 / HyperLeaf / Rices90 en local.
  Disponible et **public** (Zenodo 17397954) : scènes SAR UAVSAR du chapitre
  détection, `2-detection/data/SAR/`. `Scene4_cropped` = (500, 800, 3 pol,
  68 dates) complex64 → **204 canaux**, échelle HyperLeaf.

## Phase 1 — Le « bug » de gradient  `[x]`

Dépôt `yetanotherspdnet`, branche `fix/whitening-congruence-matrix-grad`.

- `[x] 1.1` **Il n'y a pas de bug.** L'écart signalé au §8 de la note venait de
  mon test, qui pilotait la passe arrière avec un cotangent non symétrique. Avec
  un cotangent symétrique — le seul cas réalisable dans un réseau à
  représentations SPD — manuel et autograd coïncident à 2e-14 et 2e-16.
  Dérivation à l'appui : $\partial_M\langle G,MXM\rangle = 2\,\mathrm{sym}(GMX)$
  ssi $G$ est symétrique. Note corrigée.
- `[x] 1.2` La précondition n'était ni documentée ni testée (tous les tests de
  backward passent par `torch.norm`, dont le cotangent est symétrique par
  construction). Documentée sur les deux `backward`.
- `[x] 1.3` Test à cotangent symétrique arbitraire ajouté pour `Whitening` et
  `CongruenceSPD` — strictement plus fort que celui de la norme. 3797 tests au
  vert. Commit `932156c`.
- `[x] 1.4` Empaquetage réparé (`packages.find` au lieu d'un unique paquet),
  vérifié par installation dans un venv neuf. Commit `978cba8`.
- `[ ] 1.5` Ouvrir les deux PR en amont.

## Phase 2 — Étude ReEig  `[~]`

`reeig_spectrum/`, voir son README. Sert `rem:spdnet-covpool-regime` et
`rem:spdnet-reeig-retrecissement`, remplace la figure GPR abandonnée (§5 de la
note). Réutilise `ReEig` de `yetanotherspdnet` et les chargeurs de
`spdnet-datasets`.

*(Les données SAR envisagées d'abord sont abandonnées : une covariance
polarimétrique n'est pas une CovPool, ça n'éclairait ni l'un ni l'autre des deux
renvois.)*

- `[x] 2.1` `common.py` : CovPool de `eq:spdnet-covpool` écrite telle quelle,
  covariance vraie à décroissance géométrique (`random_SPD` ne convient pas —
  spectre uniforme, un seul écrêtage par matrice quel que soit $\varepsilon$),
  et le facteur de Loewner en forme close $1/\lambda_{\min}$.
- `[x] 2.2` `main.py`, simulé : balayage à deux axes (ratio d'échantillonnage ×
  décroissance spectrale). Résultat : au ratio 3 et à décroissance $10^6$, ReEig
  écrête 35 % du spectre et divise le facteur de Loewner par 170 ; à
  décroissance $10^2$ elle ne fait rien ; sous le ratio 1 la matrice est
  singulière et LogEig n'est pas défini.
- `[x] 2.3` **La borne** : $\max_{ij}|\mathbf{G}_{ij}| = 1/\lambda_{\min}$, donc
  ReEig la plafonne à $1/\varepsilon$. Vérifié exactement.
- `[x] 2.4` Export `\prov` + sidecar, largeur 355 pt.
- `[~] 2.5` `real_data.py` écrit et auto-testé, **pas exécuté** (données
  absentes de cette machine). À lancer sur une machine qui les a, `--device cuda`.
- `[ ] 2.6` Une fois les chiffres réels connus : rédiger le bloc de
  `sec:spdnet-covpool` et poser `rem:spdnet-reeig-retrecissement`.

## Phase 2 bis — Ordre de l'équivalence projavg/rlavg  `[x]`

`stiefel_aggregation/`, voir son README. Aucune donnée, 2 s.

- `[x]` Ordre mesuré à **2,98–3,00 sur neuf configurations** (trois géométries
  de Stiefel × trois nombres de clients). La proposition du chapitre est vraie
  et conservatrice.
- `[ ]` Trancher : corriger l'énoncé en $O(\varepsilon^3)$, ou garder
  $O(\varepsilon^2)$ en disant que la borne est atteinte avec marge.

## Phase 2 ter — Volet coût du §batch-norm  `[x]`

`batchnorm_cost/`, voir son README. Aucune donnée. **Réécriture** : aucun dépôt
ne contient de mesure de temps ou de mémoire.

- `[x]` Mémoire mesurée par `saved_tensors_hooks` plutôt que par
  `cuda.max_memory_allocated` : c'est exactement ce que
  `prop:spdnet-grad-geo` prédit (les $\Niterm$ itérées retenues), c'est exact,
  attribuable, et indépendant du matériel. Le pic d'allocateur est relevé en
  plus sous CUDA.
- `[x]` Les trois balayages de l'article (taille, batch, profondeur) : rapport
  mémoire **croissant sur les trois axes**, 2,42× à taille 512 jusqu'à 3,12× à
  profondeur 32, et toujours plus élevé pour la géométrique que pour GAH.
- `[x]` Quatrième balayage, hors article, sur $\Niterm$ : 1,37× à une itération,
  2,91× à vingt. C'est le mécanisme derrière les trois autres.
- `[x]` Temps : **comparables à quelques pour cent près**, dans les deux sens.
  L'argument est la mémoire et la robustesse, pas la vitesse — dit franchement
  dans le README, comme le bloc `% TODO` le demande.
- `[ ]` Relancer les temps sur la machine de mesure (les mémoires sont
  déterministes, les temps non).

## Phase 2 quater — Cinq moyennes dans la figure du ch. 3  `[x]`

Ajout 5 du bloc `TODO decision`, fait pour le coût d'un seul run.

- `[x]` `1-context/riemann_moyennes/main.py` étendu aux moyennes harmonique et
  GAH (calculée comme milieu de géodésique, donc la définition elle-même).
  Arithmétique et harmonique encadrent le nuage (det 1,32 et 0,86 contre 1,03),
  GAH atterrit près de Fréchet (1,068, distance 0,120 contre 0,081) sans
  itération.
- `[x]` Rejoué sous qanat (run 39), réenregistré, resynchronisé.
- `[x]` Légende et texte de `3-Riemann.tex` mis à jour (ils annonçaient trois
  moyennes) et renvoi ajouté vers `prop:spdnet-moyennes-frechet`.

## Phase 2 quinquies — `just build` réparé  `[x]`

Échec **préexistant** rencontré en vérifiant la figure, reproduit sur l'arbre
non modifié : l'externalisation tikz mourait sur `figure0`.

- `[x]` Cause : la garde de `\joinedpaper` comparait `\jobname` à
  `\tikzexternalrealjob`, or la bibliothèque `external` redéfinit `\jobname`
  pour rendre le nom du document principal pendant une passe figure. Les deux
  valaient donc toujours `dissertation`, le test passait *tout le temps*, et
  `\includepdf` s'exécutait là où `\includegraphics` est neutralisé.
- `[x]` Remplacée par `\tikzifexternalizing`, la macro publique de PGF prévue
  pour cela. Vérifiée en isolation sur les deux passes.

## Phase 3 — Restitution  `[x]`

- `[x] 3.1` `NOTE-reproduction.md` à jour : §8 corrigé (il n'y avait pas de
  bug), §8 bis ajouté (verdict MPS), §4 et §5 mis à jour par les mesures.
- `[x] 3.2` `docs/docs/chapters/4-deeplearning.md` écrit.
- `[x] 3.3` Descripteurs qanat des deux expériences.

## Bloqué sur une décision d'Ammar

- **Pousser les deux commits de `yetanotherspdnet`** (correctif d'empaquetage +
  test de contrat) en amont : action sortante, pas faite sans accord. Tant
  qu'elle ne l'est pas, `yetanotherspdnet` ne peut pas devenir une dépendance
  du dépôt, donc `uv run` ne suffit pas et la chaîne qanat ne peut pas être
  lancée telle quelle (`PYTHONPATH` en attendant).
- **Lancer `real_data.py`** sur une machine qui a les jeux de données.
- Les cinq points du §9 de la note.

## Hors périmètre (décidé dans la note)

GPR, les 3 jeux réels de la batch-norm, l'EEG fédéré. La figure Wishart se
relance telle quelle depuis `eusipco_2026` et n'est donc pas réécrite ici.
