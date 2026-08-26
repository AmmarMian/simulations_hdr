# Chapitre `ch:spdnet` — que reproduire, et à quel prix

Note d'analyse, 2026-08-26. Répond aux trois blocs `% TODO decision` /
`% TODO texte + experiences` de `Chapters/6-SPDNet.tex`
(§`sec:spdnet-covpool-resultats`, §`sec:spdnet-batchnorm-resultats`,
§`sec:spdnet-federe-resultats`).

Tout ce qui est chiffré ci-dessous a été **mesuré en session** sur ce Mac (CPU,
float64), pas lu dans les articles. Les dépôts ont été clonés et exécutés.

---

## 1. Recommandation en une ligne par section

| Section | Figures de l'article | Décision proposée | Coût |
|---|---|---|---|
| §`covpool` (GPR, Jafuno) | 3 figures | **Ne rien rejouer.** Citation, pdf joint. | — |
| §`batchnorm` volet 1 (coût) | 3 figures | **Réécrire** (≈150 lignes, code inexistant dans les dépôts). | 1 j + un GPU |
| §`batchnorm` volet 2 (3 jeux réels) | 3 tableaux | **Ne rien rejouer.** Citation. | (semaines GPU) |
| §`batchnorm` — **figure nouvelle** | *n'existe pas* | **À FAIRE EN PRIORITÉ.** Wishart / Wishart inverse. | **2 min 32 s, CPU** |
| §`fédéré` (EEG, Pautrel) | 1 figure + 1 tableau | **Ne rien rejouer.** Citation. | — |
| §`fédéré` — **figure nouvelle** | *n'existe pas* | **À FAIRE.** Ordre de `prop:spdnet-federe-equivalence`. | **≈30 lignes, 2 s, CPU** |

La logique est celle du chapitre détection : on ne rejoue une figure que si
elle porte une étape de l'argument *du mémoire*. Or les deux figures les plus
utiles à ce chapitre **ne sont dans aucun article** — ce sont exactement les
deux que la convention `\prov` peut porter, et les deux les moins chères.

---

## 2. Carte des dépôts

Cinq dépôts, tous publics sauf `sigpro_2026`, tous clonés et inspectés.

| Dépôt | Rôle | État |
|---|---|---|
| `yetanotherspdnet` | couches SPD (BiMap, ReEig, LogEig, **BatchNormSPDMean**), 5 géométries, Stiefel | public, testé, **installation cassée** (§6) |
| `spdnet-datasets` | chargeurs réels + **générateurs synthétiques** (Wishart / Wishart inverse) | public |
| `spdnet-training` | Lightning + Hydra + Optuna | public, lourd |
| `eusipco_2026` | **ARMAGNAC** = GAH adaptative + module `simulation` | public |
| `sigpro_2026` | version journal : les 7 grid searches | privé |

**Correction à ta liste** : le mappage est l'inverse de ce que tu m'as indiqué.
`eusipco_2026` *est* le dépôt ARMAGNAC (`--wishart-inverse` étiquette la méthode
adaptative « ARMAGNAC » dans sa sortie), et `sigpro_2026` est la version journal.
Donc `EUSIPCO26a` → `eusipco_2026`, `PREP26b` → `sigpro_2026`. Rien à changer
dans le `.tex`, mais à ne pas inverser dans le bloc « code disponible » du
§`sec:spdnet-batchnorm-gradmoyennes`, qui parle aujourd'hui d'un dépôt
`on_batchnormalization_for_spdnet` — **ce nom n'existe pas** dans
l'organisation ; le bon renvoi est `sigpro_2026` (+ `yetanotherspdnet` pour les
couches).

**Le dépôt fédéré (`EUSIPCO26b`, Pautrel) n'existe pas** dans l'organisation.
C'est ce qui ferme la question §`fédéré` : rien à rejouer, faute de code.

---

## 3. La figure à faire : Wishart contre Wishart inverse

C'est le point important de cette note.

`prop:spdnet-moyennes-frechet` dit que chaque moyenne est l'estimateur d'un
modèle (arithmétique ↔ Wishart, harmonique ↔ Wishart inverse, GAH ↔ KL
symétrisée). Le bloc `% TODO texte --- LE POINT CENTRAL` demande d'en faire
« un énoncé de modélisation, pas une curiosité numérique ». Aujourd'hui, le seul
appui de cet énoncé serait les tableaux de F1 sur HDM05 / HyperLeaf / Rices90 :
un argument **indirect**, sur des données dont on ne connaît pas la loi.

Or `eusipco_2026` contient une simulation qui teste l'énoncé **directement** :
on tire les données d'un Wishart, puis d'un Wishart inverse, et on regarde
quelle moyenne gagne. Cette expérience **n'apparaît dans aucune des figures que
le chapitre prévoit**.

Grille `wishart-inverse` lancée intégralement en session (50 configurations :
5 moyennes × 5 graines × {Wishart, Wishart inverse}, matrices 64×64, 3 classes,
150 époques max) — **2 min 32 s sur 8 cœurs CPU, sans GPU et sans aucune
donnée** :

| Moyenne | données Wishart | données Wishart inverse |
|---|---|---|
| Arithmétique (KL gauche) | **90,6 % ± 4,2** | 55,3 % ± 22,1 |
| Harmonique (KL droite) | 46,1 % ± 8,5 | **88,9 % ± 5,7** |
| GAH (KL symétrisée) | 76,7 % ± 5,6 | 81,4 % ± 5,9 |
| ARMAGNAC (GAH adaptative) | 86,7 % ± 2,3 | 87,2 % ± 6,2 |
| Géométrique (affine invariante) | 90,0 % ± 4,9 | 88,1 % ± 5,3 |

Lecture : **la moyenne accordée au modèle gagne, et la moyenne opposée
s'effondre** — 90,6 → 46,1 pour l'arithmétique quand on retourne le modèle,
55,3 → 88,9 pour l'harmonique. L'écart-type de 22,1 sur l'arithmétique en
Wishart inverse dit la même chose autrement : sous mauvais modèle,
l'apprentissage n'est même plus reproductible d'une graine à l'autre. Les
moyennes symétriques (GAH, ARMAGNAC) et la moyenne géométrique, elles, ne
perdent presque rien en changeant de modèle.

C'est la phrase que le chapitre cherche à produire, et elle devient **mesurée**
au lieu d'être affirmée.

**Deux réserves d'honnêteté**, à respecter dans le texte :

1. Sur ces données, la **moyenne géométrique est la meilleure des cinq à peu
   près partout** (90,0 / 88,1). La simulation ne dit donc *pas* « GAH bat la
   géométrique » — elle dit « la moyenne accordée au modèle gagne, les moyennes
   symétriques sont robustes au modèle ». C'est un énoncé plus fort et plus
   défendable. L'argument « la géométrique n'est jamais la meilleure » reste ce
   qu'il était : un fait **des trois jeux réels**, à laisser dans le volet 2, et
   à ne surtout pas mélanger avec cette figure. Bien séparées, les deux se
   renforcent : la simulation établit le mécanisme, le réel montre que sur ces
   données-là l'hypothèse gaussienne n'est pas la bonne.
2. Les données sont un Wishart / Wishart inverse dont les matrices d'échelle
   diffèrent par une perturbation de 25 % : la séparation entre classes est un
   réglage, pas une propriété. Le dire.

**Ajout suggéré, gratuit** : la grille tourne aujourd'hui sur un seul jeu
(64×64, df = 64). Faire varier `df` de 64 à 500 donnerait un second panneau —
quand df grandit, le Wishart se concentre et l'écart entre moyennes doit se
refermer. Le mécanisme deviendrait visible comme un *gradient*, pas comme deux
points. C'est un `--custom` du même runner, quelques minutes de plus.

**ARMAGNAC dans le chapitre.** La GAH adaptative est aujourd'hui **absente du
`.tex`** : `prop:spdnet-moyennes-frechet` liste cinq moyennes,
`tab:spdnet-moyennes` cinq lignes, et `EUSIPCO26a` n'est même pas cité dans
l'encadré `contributions` d'ouverture (seulement dans le commentaire final sur
les pdf joints). Si tu prends cette figure, ARMAGNAC arrive avec elle — et il
arrive bien : c'est la seule méthode qui n'a pas à choisir son modèle
*a priori*, ce qui est la conclusion naturelle du tableau ci-dessus. Une
définition et une ligne de tableau suffisent.

---

## 4. La seconde figure à faire : l'ordre de l'équivalence projavg / rlavg

`prop:spdnet-federe-equivalence` affirme que les deux agrégations coïncident à
`O(ε²)` près. Le bloc `% TODO texte` demande de justifier cela « en deux lignes
de calcul » et de renvoyer aux courbes EEG « qui montrent des trajectoires
superposées ». Des trajectoires superposées ne mesurent pas un ordre.

Vérification directe, sans données et sans entraînement — tirer K poids locaux à
distance ε de l'itérée globale sur `St(40, 20)`, agréger des deux façons,
mesurer l'écart (`stiefel_projection_polar` et
`stiefel_projection_tangent_orthogonal` de `yetanotherspdnet` sont exactement
`polarf` et `Lift` de `def:spdnet-lift`) :

```
     eps     ||projavg - rlavg||_F     pente
    1e+00           9,12e-03
    1e-01           9,59e-06            2,98
    1e-02           9,44e-09            3,01
    1e-03           1,01e-11            2,97
    1e-04           1,44e-14            2,85   ← plancher machine
```

**Pente 3, pas 2.** La proposition est vraie et *conservatrice* : l'écart est en
`O(ε³)`. À 1 % de dispersion entre clients, les deux schémas diffèrent de 1e-8
en norme de Frobenius — c'est-à-dire qu'ils sont identiques. La recommandation
du chapitre (« `projavg` est recommandée, constantes plus faibles, pas besoin de
garder l'itérée précédente ») cesse d'être un arbitrage et devient un choix sans
contrepartie.

30 lignes, 2 secondes, aucune donnée. À balayer sur `(d₀, d₁, K)` avant
d'énoncer l'ordre 3 : une seule géométrie a été testée. Si l'ordre 3 tient, il
faut soit corriger l'énoncé de la proposition, soit dire dans le texte que la
borne `O(ε²)` est atteinte avec marge.

---

## 5. Ce qu'on ne reproduit pas

**§`covpool` — GPR (Jafuno).** Ta position : les résultats GPR ne te semblent
pas fiables, et c'est tout ou rien. C'est **rien**, pour trois raisons qui
s'additionnent : les données Geolithe ne sont pas distribuables (donc aucune
figure ne serait rejouable par un lecteur, ce qui vide la macro `\prov` de son
sens) ; le code n'est pas dans cette organisation (`anotherspdnet`, cité au
§`sec:spdnet-bilan`, est un autre dépôt) ; et rejouer une chaîne
ResNet + CovPool + SPDnet sans confiance dans le code d'origine, c'est réécrire,
pas reproduire. La convention du mémoire couvre ce cas : figure reprise d'un
article → citation. La section est courte par construction, cela ne coûte rien.

*Substitut honnête, si tu veux quand même une figure `\prov` dans cette
section* : **fait**, voir `reeig_spectrum/`. `rem:spdnet-covpool-regime` avance
que $\Npix/\Nfilter \approx 3$ place `CovPool` en plein régime dimensionnel du
§`context-covariance` ; l'expérience le mesure sans les données GPR, et donne
au passage la borne $\max_{ij}|\mathbf{G}_{ij}| \le 1/\varepsilon$ qui fait de
`rem:spdnet-reeig-retrecissement` un énoncé et non une analogie. Un script
jumeau `real_data.py` porte la même mesure sur HDM05 / HyperLeaf / Rices90, à
lancer sur une machine qui les a.

**§`batchnorm` volet 2 — les trois jeux réels.** Sept grid searches séquentiels
(`scripts/01_…` à `07_…`), chacun un `--multirun` Hydra sur 5 graines avec
`launcher=gpu_sweep`, et un `DATA_ROOT` à constituer (HDM05 sur demande de
licence, HyperLeaf CVPR 2024, Rices90). Les scripts consignent d'ailleurs les
meilleurs hyperparamètres **en dur, à la main** (`# UPDATE THESE VALUES!`),
donc l'enchaînement n'est pas automatisable tel quel. Sans intérêt pour le
mémoire : les chiffres du volet 2 sont déjà dans le pdf joint, et l'argument
qu'ils portent (autograd échoue sur HyperLeaf et Rices90) est un fait rapporté,
pas une figure.

**§`fédéré` — EEG.** Pas de dépôt. Les jeux MOABB sont publics et
`projavg`/`rlavg` font vingt lignes, mais il faudrait écrire tout le harnais
fédéré (150 tours × 5 ou 53 clients × 2 époques). Hors budget, et sans
contrepartie : le tableau final du chapitre est un tableau de chiffres, il se
cite.

---

## 6. Volet coût (§`batchnorm` volet 1) — à réécrire

**Aucun des cinq dépôts ne contient de code de mesure de temps ou de mémoire.**
Les trois figures coût/mémoire de `PREP26b` ont été produites par du code qui
n'a pas été versionné. Il n'y a donc pas de « rejouer » possible : c'est une
réécriture.

La bonne nouvelle est qu'elle est facile, parce que le commutateur existe déjà
dans la bibliothèque :

```python
BatchNormSPDMean(n_features, mean_type=..., use_autograd=True|False)
```

`mean_type ∈ {affine_invariant, log_euclidean, arithmetic, harmonic,
geometric_arithmetic_harmonic, adaptive_geometric_arithmetic_harmonic}`, et
`random_SPD(n_features, n_matrices, cond=1e5)` fournit exactement les entrées
mal conditionnées du protocole. Les deux chemins tournent, vérifié :

```
affine_invariant                autograd=False  26,7 ms   autograd=True  13,2 ms
geometric_arithmetic_harmonic   autograd=False   2,7 ms   autograd=True   2,6 ms
arithmetic                      autograd=False   0,7 ms   autograd=True   0,5 ms
harmonic                        autograd=False   1,2 ms   autograd=True   1,5 ms
log_euclidean                   autograd=False   2,7 ms   autograd=True   2,6 ms
```
(n = 32, N = 32, cond = 1e5, CPU — un seul passage avant + arrière.)

Cela confirme le message que le bloc `% TODO` demande de « dire franchement » :
**le gain est en mémoire, pas en temps**. Sur CPU la formule manuelle est même
deux fois plus lente que l'autograd pour la moyenne géométrique. La figure doit
donc être une figure de mémoire, avec le temps en second panneau et sans le
survendre.

Deux réserves pratiques :

- la mesure mémoire de l'article est `torch.cuda.max_memory_allocated` : **il
  faut un GPU**. À défaut, le substitut mesurable sur CPU est la *taille du
  graphe d'autograd* (nombre de tenseurs retenus × leur taille), qui est la
  quantité que la proposition explique — l'autograd conserve les $\Niterm$
  itérées de `eq:spdnet-geometrique-iteration`. C'est même plus parlant qu'un
  nombre de mégaoctets, parce que ça se prédit ;
- **contrainte torche du chapitre** : ce chapitre est le seul du mémoire à ne
  pas passer par `hdrlib.core.backend`. Les expériences de `3-deeplearning/`
  sont en PyTorch pur, sur `yetanotherspdnet`. Le harnais commun
  (`make_mc_parser`, `MCResultExporter`, `--export-path`) reste applicable et
  doit l'être, seul le cœur de calcul change.

---

## 7. Pièges d'installation (vérifiés, pas supposés)

1. **`yetanotherspdnet` ne s'installe pas correctement.** Son `pyproject.toml`
   déclare `[tool.setuptools] packages = ["yetanotherspdnet"]` : les
   sous-paquets `functions/`, `nn/`, `random/`, `spd_geometries/` ne sont
   **pas** copiés. Après un `pip install`, `import yetanotherspdnet` échoue sur
   un faux message d'import circulaire. Correctif d'une ligne, à remonter au
   dépôt :
   ```toml
   [tool.setuptools.packages.find]
   where = ["src"]
   ```
   (c'est déjà ce que fait `spdnet-datasets`.) En attendant :
   `PYTHONPATH=…/yetanotherspdnet/src`.
2. **Les trois dépôts d'article pointent vers `dev_yetanotherspdnet`**, qui
   renvoie 404. `uv pip install -e .` échoue donc partout. Repointer sur le
   dépôt public.
3. `spdnet-datasets` importe `yetanotherspdnet` sans le déclarer en dépendance,
   et son `__init__` charge les chargeurs réels (donc `tifffile`, `spectral`)
   même quand on ne veut que `synthetic`.
4. Une fois ces trois points contournés, la simulation §3 tourne **sans
   `spdnet-training`, sans Lightning, sans Hydra et sans `DATA_ROOT`** :
   `SimpleSPDNet` (BatchNorm → LogEig → Linear) et son entraîneur sont locaux à
   `eusipco_2026`. C'est ce qui rend cette figure bon marché.

---

## 8. L'anomalie annoncée ici n'en était pas une — corrigé le 2026-08-26

Une première version de cette note signalait un désaccord de 2 à 9 % entre les
gradients manuel et automatique de `whitening` / `congruence_SPD` sur leur
argument matriciel, et le donnait comme un bug bloquant. **C'était mon test qui
était faux.**

J'avais pondéré la perte par une matrice non symétrique, ce qui fabrique un
`grad_output` non symétrique. Avec un gradient amont **symétrique** — le seul
cas qui se produise dans un réseau dont chaque couche sort une SPD — les deux
chemins coïncident à la précision machine :

| | ∂/∂données | ∂/∂matrice, G quelconque | ∂/∂matrice, **G symétrique** |
|---|---|---|---|
| `whitening` | 2,1e-16 | 1,3e-01 | **2,3e-14** |
| `congruence_SPD` | 1,6e-16 | 5,7e-01 | **2,0e-16** |

La dérivation le confirme :
$\partial_M\langle G, MXM\rangle = GMX + XMG = 2\,\mathrm{sym}(GMX)$ **si et
seulement si** $G$ est symétrique, et c'est exactement ce que le code calcule.
Le backward manuel ne propage que la partie symétrique à `M`, ce qui est correct
puisque `M` est contrainte SPD. Il n'y a pas de bug, et rien ne bloque la
figure §3.

**Ce qui reste vrai et utile** : la précondition n'était écrite nulle part, et
aucun test ne l'exerçait — tous les tests de backward pilotent la passe par
`torch.norm(...)`, dont le cotangent $Y/\lVert Y\rVert$ est symétrique parce que
$Y$ l'est. Elle est désormais documentée sur les deux `backward` et verrouillée
par un test à cotangent symétrique arbitraire (branche
`fix/whitening-congruence-matrix-grad`, 3797 tests au vert).

Leçon pour moi : un écart entre deux implémentations n'est un bug que si
l'entrée qui le révèle peut se produire.

---

## 8 bis. Verdict MPS (torch-metal), mesuré le 2026-08-26

**Inutilisable pour ce chapitre.** Trois raisons cumulatives :

- MPS n'a **pas de float64**, or `yetanotherspdnet` est en float64 partout
  (`dtype=torch.float64` par défaut sur chaque couche) ;
- `torch.linalg.eigh` **n'est pas implémenté** sur MPS — c'est l'opération de
  ReEig, LogEig, `sqrtm`, et des cinq moyennes ;
- `torch.linalg.svd` y retombe aussi sur le CPU, donc la rétraction polaire du
  §fédéré également.

Avec `PYTORCH_ENABLE_MPS_FALLBACK=1`, `eigh(256×64×64)×10` prend 0,388 s contre
**0,364 s en CPU pur** : aucun gain, et un aller-retour mémoire par opération.
Les scripts de `3-deeplearning/` prennent donc `--device cpu|cuda` et refusent
`mps` explicitement plutôt que de le dégrader en silence.

## 9. Ce qu'il reste à trancher

1. La figure Wishart / Wishart inverse entre-t-elle dans le chapitre ? (Ma
   recommandation : oui, c'est le meilleur rapport qualité/prix du mémoire.)
   Si oui : ARMAGNAC entre avec elle, et `EUSIPCO26a` doit rejoindre l'encadré
   `contributions` d'ouverture.
2. La figure d'ordre `projavg`/`rlavg` — et faut-il corriger l'énoncé `O(ε²)`
   de `prop:spdnet-federe-equivalence` en `O(ε³)` ?
3. Le volet coût : GPU disponible, ou substitut « taille du graphe d'autograd »
   sur CPU ?
4. Le substitut régime dimensionnel du §`covpool` (§5) : à faire, ou section
   sans figure `\prov` du tout ?
5. Corriger le renvoi `on_batchnormalization_for_spdnet` →
   `sigpro_2026` + `yetanotherspdnet` dans le bloc `% TODO texte` du
   §`sec:spdnet-batchnorm-gradmoyennes`.
