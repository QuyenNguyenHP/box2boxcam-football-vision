# Inventaire des données PitchLens — 10 octobre 2026

## Résumé

Nous avons **6 769 images annotées dans quatre exports figés**, dont **2 220
images dans la nouvelle filière Luna + vidéos personnelles**. Les 4 549 autres
proviennent de l'ancien dataset TV Infactory.

Ces nombres décrivent le stock disponible, **pas un dataset fusionné déjà
configuré pour entraîner**. Les anciens splits sont conservés ; les données
personnelles sont exportées séparément en train et test.

Un nouveau contrôle local des images et annotations a été effectué le 10 octobre :
aucun problème de format, aucune image manquante et **6 769 empreintes JPEG
SHA-256 distinctes**. Cela vérifie la structure et les doublons exacts, pas
l'exactitude sémantique de chaque boîte ni l'indépendance des scènes.

## Les quatre exports disponibles

| Export | Origine et état de la revue | Images | Avec ballon | Sans ballon | Train / val / test |
|---|---|---:|---:|---:|---|
| `v002` | Infactory, images de retransmissions TV triées/corrigées | 4 549 | 3 246 | 1 303 | 3 048 / 670 / 831 |
| `luna_v001` | Première compilation vidéo, Luna puis revue humaine | 1 120 | 875 | 245 | 768 / 176 / 176 |
| `lou_mistrau_train_v001` | Tes trois vidéos destinées au train, revues aux seuils confirmés | 808 | 787 | 21 | 808 / — / — |
| `lou_mistrau_test_v001` | Ta courte vidéo du 30 septembre, revue sans seuil | 292 | 264 | 28 | — / — / 292 |
| **Total du stock** | Les exports ne sont pas fusionnés | **6 769** | **5 172** | **1 597** | **4 624 / 846 / 1 299** |

Pour la nouvelle filière uniquement : **2 220 images**, dont **1 926 positives
et 294 négatives**. Les splits existants représentent 1 576 images de train,
176 de validation et 468 de test, avant toute décision de composition d'expérience.
Les 292 images personnelles restent le benchmark ciblé sur ton usage.

### 1. Ancien dataset TV — `v002`

- Source : Infactory Ball Tracking, révision et provenance dans `build_report.json`.
- 7 152 images sources ; 2 603 exclusions ; 4 549 images retenues.
- 246 décisions de correction de boîte, 92 décisions négatives, 1 298 validations
  explicites. Les 2 913 images sans décision conservent les labels d'origine.
- Les séquences sont séparées entre les splits ; zéro doublon JPEG exact dans
  le contrôle actuel.
- Ce dataset constitue une réserve utile et le point de comparaison historique,
  mais son domaine TV est éloigné d'un téléphone au bord d'un terrain amateur.
  « Nettoyé » ne signifie pas que chaque label conservé par défaut est parfait.
- Les restrictions de source restent celles consignées dans
  `configs/data_sources.toml` ; aucune nouvelle publication ou modification de
  licence n'a été effectuée pour cet inventaire.

### 2. Première compilation Luna — `luna_v001`

- 1 232 frames initiales, échantillonnées une sur quatre.
- 68 exclusions humaines et 44 frames écartées autour des transitions.
- 1 120 images finales ; 10 groupes de matchs identifiés visuellement.
- Les groupes sont exclusifs entre train, validation et test **selon cette
  identification visuelle**, pas selon des identifiants de matchs certifiés.
- Les boîtes viennent de Luna, avec les corrections, brouillons soumis et
  propositions inchangées acceptées à ta demande lors de la revue.
- Point de vigilance conservé dans le rapport d'origine : **49 images sans
  réponse Luna ont été soumises avec un label vide**. Si l'on renforce la qualité
  de cette ancienne version, ce sont des négatifs à recontrôler en priorité.
- Statut de diffusion consigné : expérimentation privée, droits de
  redistribution non clarifiés. Ce travail ne change pas ce statut.

### 3. Tes vidéos — nouveau train `lou_mistrau_train_v001`

| Vidéo | Règle confirmée | Retenues | Avec ballon | Sans ballon | Exclusions |
|---|---|---:|---:|---:|---:|
| 30 septembre, longue — 19 h 38 | Confiance ≥ 0,86, revue confirmée | 427 | 422 | 5 | 1 |
| 7 octobre — 20 h 40, portrait | Confiance ≥ 0,88 + corrections soumises sous le seuil | 250 | 235 | 15 | 0 |
| 7 octobre — 20 h 47, paysage | Confiance ≥ 0,88 + corrections soumises sous le seuil | 131 | 130 | 1 | 4 |
| **Total** | | **808** | **787** | **21** | **5** |

Les seuils sont inclusifs : une image à 0,86 est gardée pour la longue vidéo.
Une annotation humaine soumise a priorité sur Luna, y compris lorsqu'elle
supprime une boîte ou corrige une image sous le seuil. Les exclusions passent
avant toute autre règle. **18 images corrigées/validées sous le seuil** sont
ainsi conservées dans les deux vidéos du 7 octobre.

Parmi les 808 images retenues, 55 ont une annotation soumise et 753 gardent la
proposition Luna inchangée, après ta confirmation de revue de ces plages de
confiance. Leur provenance est distinguée dans le manifeste ; on ne prétend
pas que ces 753 images ont individuellement reçu Submit dans l'outil.

Résolutions natives : 250 images portrait en 1080 × 1920 et 558 images paysage
en 1920 × 1080. Aucun redimensionnement destructif n'a été appliqué à cet export.

Le dataset n'a pas encore de validation personnelle distincte : son YAML
contient uniquement le train. Il faut définir une validation indépendante
avant de configurer un entraînement complet, **sans transformer le test en val**.

### 4. Ton benchmark réel — `lou_mistrau_test_v001`

- Courte vidéo du 30 septembre, à 19 h 41 ; 380 frames proposées, une sur quatre.
- 88 exclusions humaines ; **292 images retenues sans filtre de confiance**.
- 264 images avec ballon, 28 négatives ; **279 boîtes** au total.
- 11 images contiennent plusieurs boîtes humaines. Elles n'ont pas été
  supprimées ou réduites automatiquement à une seule boîte. Pour un benchmark
  « une seule meilleure détection », il faudra préciser la règle de référence
  sur ces images afin de ne pas confondre détection de tous les ballons et
  choix du ballon actif.
- Les 92 annotations soumises retenues et les 200 propositions acceptées
  inchangées après ta revue sont distinguées dans le manifeste.
- Ce test est inchangé et reste réservé à l'évaluation.

**Limite importante :** la longue vidéo du 30 septembre est dans le train et
la courte dans le test, selon ton choix. Les images ne sont pas identiques et
les clips sont distincts, mais **le même match/date/terrain est partagé**.
Cela mesure la performance sur un autre extrait de cette situation, pas la
généralisation à un match complètement nouveau. Les frames voisines ne sont
pas non plus des observations indépendantes.

## Ce qui n'est pas compté comme données propres

- **782 propositions Luna** sur les six shorts YouTube/TikTok. Le projet de
  revue possède actuellement zéro annotation humaine soumise et zéro brouillon.
- Les **365 propositions positives à confiance ≥ 0,95** sont un sous-ensemble
  de ces mêmes 782 images, pas 365 images supplémentaires. Elles restent des
  candidates ; le seuil seul ne vaut pas validation des boîtes.
- **1 064 images personnelles sous les seuils, sans annotation soumise** :
  elles restent accessibles dans Label Studio, mais ne sont pas intégrées au train.
- **10 requêtes expirées** sur la longue vidéo : aucune boîte ni confiance
  disponible. Elles ne sont ni des négatifs ni des exclusions humaines, et
  aucune nouvelle requête payante n'a été lancée pour elles.
- `v001`, les sources brutes et les extractions de teachers sont des versions
  parentes ou des propositions ; elles ne s'ajoutent pas au total des quatre
  exports ci-dessus.
- Les autres sources explorées, comme SoccerEye, ne sont pas intégrées à ces
  exports propres.

## Où sont les fichiers et comment les utiliser

- `data/processed/v002` : ancien dataset TV et son découpage historique.
- `data/processed/luna_v001` : première compilation revue et ses splits.
- `data/processed/lou_mistrau_train_v001` : nouveau train personnel et décisions figées.
- `data/processed/lou_mistrau_test_v001` : test réel réservé.
- `docs/data_inventory_20261010.json` : contrôle mécanique et dédoublonnage exact
  effectués pour cet inventaire.

Dans le nouveau train : les images sont dans `train/images`, les annotations
YOLO dans `train/labels`, les négatifs ont un fichier de label vide. Le
`review_manifest.json` conserve aussi les exclusions et non-sélections, avec
la provenance des annotations. `missing_predictions.json` conserve les dix
expirations. Les JPEG sont liés physiquement aux images revues, sans recopier
leur contenu. Les sources et les anciens datasets n'ont pas été remplacés.

Construction reproductible du nouveau train :

```bash
python -m pitchlens.data.build_personal_train --confirm-reviewed-thresholds
```

Le constructeur refuse de remplacer une version existante. Une correction
ultérieure doit produire une nouvelle version avec `--output`.

## Prochaine étape

Définir la composition de l'expérience suivante : commencer par la filière
amateur/Luna, choisir la validation sans répartir aléatoirement les frames
voisines, et conserver les 292 images personnelles pour le benchmark final.
Le nouveau train personnel contient seulement **21 négatifs sur 808 images**
(2,6 %) : compléter les cas sans ballon et les cas difficiles pourrait aider
à traiter les faux positifs. Ensuite seulement : entraînement, comparaison
avant/après et relecture de la courte vidéo avec les prédictions.

Aucun entraînement, upload ou appel Luna payant n'a été lancé pendant cette
préparation et cet inventaire initial.

## Expérience dérivée ensuite : nano mobile à 960 px

À ta demande suivante, `data/processed/mobile_v001` compose une expérience de
**1 547 images de train et 168 images de validation** : compilation Luna revue
+ les 808 images personnelles. Ces images sont déjà dans les quatre exports
ci-dessus : **elles ne s'ajoutent pas aux 6 769 images du stock**.

Les 29 négatifs ambigus du train Luna et les 8 de sa validation sont écartés de
cet essai. Les 12 autres négatifs ambigus appartiennent à l'ancien test Luna,
qui n'est pas utilisé dans cet entraînement. Les annotations corrigées en
positif restent incluses. Les exports parents ne sont pas modifiés.

Les 168 images de validation gardent leurs groupes de matchs d'origine ; aucun
split aléatoire des frames voisines n'est ajouté. Le test personnel reste
réservé. L'entraînement YOLO26n à 960 px utilise HF en privé, avec 60 époques
maximum et une limite GPU d'une heure. Sa configuration et ses résultats se
lisent dans `notebooks/09_mobile_training.ipynb` ; le statut du job est consigné
dans `models/reports/yolo26n_mobile_v001_960_20261010_job.json`.
