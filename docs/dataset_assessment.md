# Évaluation initiale des datasets

État de l'analyse : 15 septembre 2026.

## Conclusion courte

Le meilleur point de départ disponible est **Infactory**, après nettoyage. Il
contient des séquences consécutives, beaucoup de vrais négatifs et assez de
ballons annotés pour établir une première baseline. **SoccerEye** peut compléter
l'entraînement avec des ballons beaucoup plus petits, mais ses splits publiés
ne sont pas fiables et sa licence de données n'est pas indiquée.

Aucun de ces datasets ne représente réellement la cible : un iPhone fixe au
bord d'un terrain amateur. Ils permettront d'apprendre et de construire le
pipeline, mais la validation finale devra être faite sur nos propres vidéos.

Le fait que le projet soit aujourd'hui non commercial rend utilisables les
sources réservées à la recherche ou au non-commercial. Cela ne supprime pas les
obligations d'attribution, les interdictions de redistribution, ni
l'incertitude d'une licence absente.

## Décision par source

| Source | Utilité immédiate | Décision actuelle | Raison principale |
|---|---|---|---|
| Infactory Ball Tracking | Forte | Utiliser après nettoyage | 7 152 frames temporelles, mais 75 annotations invalides et domaine TV |
| SoccerEye | Moyenne | Conserver localement, refaire les splits | Ballons très petits, mais seulement 427 boîtes et fuite entre splits |
| Roboflow Football Ball v4 | Moyenne | Attendre l'accès authentifié | 4 948 tuiles sous CC BY 4.0, issues de 1 237 images parentes |
| SoccerNet Tracking | Forte pour le tracking | Plus tard, après NDA | Très riche temporellement, lourd et limité à la recherche |
| SoccerNet v3 H250 | Potentiellement forte | Bloqué par la licence | Sous-ensemble longue distance intéressant, licence non affichée |
| Dryad Sport Ball 2026 | Incertaine | Bloqué | Petit paquet d'annotations, images/vidéos externes non récupérées |
| Captures PitchLens | Essentielle | À créer plus tard | Seule source correspondant exactement au produit visé |

Les conditions et les liens sources sont centralisés dans
`configs/data_sources.toml`.

## 1. Infactory Ball Tracking

### Ce qui est réellement présent

[La fiche Hugging Face](https://huggingface.co/datasets/infactory-ai/ball-tracking)
annonce 1 450 frames provenant de 10 clips. La révision
téléchargée `75854a5837dd3302cbb1843355ecc1a6f124f818` contient en réalité :

- 7 152 images de données ;
- 54 séquences UUID complètes et numérotées sans trou ;
- des séquences de 15 à 588 frames, médiane 82 ;
- 4 232 images avec au moins une boîte valide ;
- 2 845 images sans ballon annoté ;
- 75 images avec une annotation de surface nulle ;
- 4 298 boîtes valides au total.

Le `metadata.csv` ne décrit encore que l'ancien sous-ensemble de 1 450 images.
Il ne doit donc pas servir de catalogue du téléchargement actuel.

### Taille du ballon

En résolution native, la boîte du ballon mesure en médiane environ
`18,0 × 16,2 px`. Après redimensionnement proportionnel et letterbox vers
`640 × 640`, elle mesure environ `13,8 × 12,5 px`.

Le dixième le plus difficile descend à environ `7,5 × 6,5 px` à 640. Cela
reste difficile, mais exploitable pour une première expérience. Les deux
résolutions présentes sont `1280 × 720` et `720 × 608`.

### Problèmes de qualité

- Une séquence de 50 frames marquée `visible` est un carton-titre télévisé avec
  des boîtes `0 0 0 0`.
- Trois autres séquences contiennent 25 annotations de surface nulle, soit 75
  images invalides sur quatre séquences au total.
- 63 images ont plusieurs boîtes alors qu'une frame normale ne devrait contenir
  qu'un ballon. L'exemple à quatre boîtes contient au moins trois annotations
  visuellement erronées ; les 63 doivent donc passer en revue humaine.
- Les images proviennent de retransmissions professionnelles : angles variés,
  gros plans, ralentis, entrelacement et filigrane. Ce n'est pas le domaine
  caméra amateur recherché.

### Traitement retenu

On ne modifie jamais `data/raw/infactory_ball_tracking`. La version de travail
`data/processed/v001` :

1. exclut les 75 annotations invalides ;
2. exclut les 63 frames multi-boîtes en attendant leur correction ;
3. sépare train/validation/test par UUID complet ;
4. réserve des séquences entières au test ;
5. conserve ensemble les frames voisines d'une même séquence ;
6. conserve les 138 exclusions dans `review_queue.csv`.

Cette correction syntaxique n'a toutefois pas suffi. Une QA visuelle
stratifiée a trouvé des boîtes uniques placées sur des objets qui ne sont pas
le ballon et une frame négative où le gardien tient visiblement le ballon.

## 2. SoccerEye

### Contenu mesuré

- 1 128 images et 28 755 boîtes ;
- classe `0` ballon : 427 boîtes dans 414 images ;
- classe `1` personne : 28 328 boîtes dans les 1 128 images ;
- 714 images ne contiennent aucun ballon et deviennent donc des négatifs pour
  un détecteur uniquement consacré au ballon ;
- 911 images en `640 × 640`, 217 en `1920 × 1080`.

Le ballon mesure en médiane `6 × 9 px` en natif. Après letterbox à 640, la
médiane tombe à environ `4 × 6,5 px`. Ce dataset est donc utile pour tester le
cas réellement difficile du ballon minuscule.

### Fuite entre les splits publiés

Les noms encodent un identifiant parent Roboflow. L'audit trouve :

- 560 noms parents seulement pour 1 128 fichiers ;
- 128 parents présents une fois, 296 deux fois et 136 trois fois ;
- 53 noms parents présents dans plusieurs splits ;
- 15 préfixes de source présumés, tous présents dans au moins deux splits ;
- aucun fichier strictement identique au niveau SHA-256 entre les splits.

Ce dernier point signifie que les copies ne sont pas identiques octet par
octet, mais les noms montrent que des variantes d'une même image parente ont
été réparties entre entraînement et évaluation. Il faut reconstruire les splits
avant de croire une métrique.

La licence MIT du [dépôt SoccerEye](https://github.com/WWandP/SoccerEye)
concerne le code. Aucune licence explicite n'a été trouvée pour le dataset lié
par Google Drive. Nous pouvons l'étudier localement dans ce projet personnel,
mais nous ne le republierons pas sur Hugging Face tant que ce point n'est pas
clarifié.

## 3. Sources non téléchargées

### Roboflow Football Ball Detection v4

[La version 4 sur Roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg/dataset/4)
indique 4 948 images générées : 3 956 train, 492 validation et 500 test. Le
prétraitement découpe chacune des 1 237 images parentes en quatre tuiles, puis
les étire en `640 × 640`. Le téléchargement automatisé demande une
authentification Roboflow.

Si nous l'ajoutons, le découpage devra être effectué au niveau de l'image
parente **avant** le tuilage. Sinon, différentes zones ou variantes d'une même
image pourraient contaminer l'évaluation.

### SoccerNet Tracking

[SoccerNet Tracking](https://www.soccer-net.org/data) est particulièrement
intéressant pour apprendre le suivi dans le temps. Le benchmark annonce 201
séquences, 225 375 frames et 215 156 boîtes de ballon. Les vidéos sont
professionnelles, en `1920 × 1080`, et accessibles après signature d'un NDA
pour la recherche. Nous ne la téléchargeons pas à ce stade : elle est plus
lourde et moins directe qu'Infactory pour la première baseline.

### Dryad et SoccerNet v3 H250

Le [paquet football de Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.3bk3j9m13)
ne fait qu'environ 3,27 Mo et semble contenir les annotations, tandis que les
vidéos sources sont hébergées séparément. Le téléchargement automatique est
actuellement refusé.

Le [sous-ensemble SoccerNet v3 H250](https://zenodo.org/records/7808511)
contient 19 786 images longue distance avec des boîtes ballon/personne. Il est
prometteur, mais l'enregistrement Zenodo n'affiche pas de licence explicite et
il dérive de SoccerNet. Il reste donc bloqué jusqu'à clarification.

## Règles scientifiques pour PitchLens

1. **Un groupe source ne traverse jamais les splits.** Pour une vidéo, le
   groupe est la vidéo ou le match, jamais la frame.
2. **Le test final reste humain et propre.** Aucun pseudo-label du teacher ne
   doit être accepté comme vérité de test.
3. **Le teacher et le student partagent le train, pas le test.** Le gros modèle
   peut proposer les labels d'entraînement ; un humain tranche les cas ambigus.
4. **On mesure les petits ballons séparément.** Les résultats seront ventilés
   par taille après resize (`< 4 px`, `4–8 px`, `8–16 px`, `> 16 px`).
5. **On conserve les négatifs.** Une caméra doit savoir rester stable quand le
   ballon est caché ou hors champ.
6. **La cible décide.** Une petite collection de matchs filmés au même endroit
   qu'un futur gimbal sera le vrai test de faisabilité.

## Dataset de travail `v001` construit

Le notebook `02_dataset_preparation.ipynb` a construit une première version
conservatrice limitée à Infactory :

- 7 014 images retenues : 4 169 positives et 2 845 négatives ;
- 138 images exclues et conservées dans une file de revue ;
- 38 séquences et 4 911 images dans le train ;
- 8 séquences et 1 071 images dans la validation ;
- 8 séquences et 1 032 images dans le test ;
- aucune séquence partagée entre deux splits ;
- aucune boîte invalide ou image multi-boîtes dans la version préparée ;
- 7 014 empreintes SHA-256 uniques, donc aucun doublon exact entre les splits.

### Résultat de la QA visuelle

Le notebook affiche 20 exemples déterministes par split : quatre ballons pour
chacune des catégories `tiny`, `small`, `medium`, `large`, puis quatre négatifs.
Sur ces 60 images, cinq erreurs certaines ont été consignées : quatre faux
positifs et un faux négatif.

La validation visuelle de `v001` est donc **échouée**, même si sa validation
mécanique est réussie. Le notebook produit une file de 700 images prioritaires :

- 508 dans le train ;
- 48 dans la validation ;
- 144 dans le test ;
- 589 grandes boîtes à vérifier ;
- 86 négatifs isolés entre deux frames positives ;
- 29 sauts brusques de position ou de surface, avec quelques recouvrements
  entre les règles ;
- les cinq erreurs déjà confirmées visuellement.

Ces signaux ne sont pas des décisions automatiques. Une grande boîte peut être
un vrai gros plan et un saut peut correspondre à un changement de caméra. Ils
servent à ordonner la revue humaine dans
`data/processed/v001/qa_review_queue.csv`.

Avant le baseline nano, il faut nettoyer au minimum les 192 candidats de
validation et de test, puis générer `v002` sans modifier `raw` ni `v001`.
SoccerEye viendra dans une expérience séparée afin de mesurer son apport sur les
très petits ballons sans masquer les défauts d'Infactory.

Pour la revue effective, `03_dataset_review.ipynb` fusionne ces 700 candidats
avec les 138 annotations déjà exclues. La file humaine complète contient donc
838 images : 600 train, 57 validation et 181 test. Les compteurs précédents de
48 validation et 144 test concernent uniquement les annotations qui avaient été
acceptées dans `v001` ; les 37 exclusions validation/test s'ajoutent dans
l'interface de correction.

### Extension à une revue exhaustive

La file de 838 candidats ci-dessus était une sélection, pas un contrôle exhaustif.
Après les erreurs relevées pendant la revue, `03_dataset_review.ipynb` affiche
désormais les 7 152 images sources d'Infactory, y compris les négatifs sans signal
automatique et les exclusions initiales. Les décisions précédentes sont conservées.
La file complète contient 1 069 images test, 1 080 validation et 5 003 train,
ordonnées par split, séquence et frame. Les signaux servent à expliquer les risques,
plus à restreindre la couverture. SoccerEye reste séparé.

## v002 construite — 16 septembre 2026

Règle validée par l'utilisateur : absence de décision = conservation du label
d'origine. Plusieurs vrais ballons sont autorisés. Les corrections explicites
remplacent le label ; les exclusions individuelles et groupées retirent l'image.

| Split | Images | Positives | Négatives | Boîtes | Séquences |
|---|---:|---:|---:|---:|---:|
| Train | 3 048 | 2 113 | 935 | 2 114 | 30 |
| Validation | 670 | 494 | 176 | 494 | 7 |
| Test | 831 | 639 | 192 | 651 | 8 |
| Total | 4 549 | 3 246 | 1 303 | 3 259 | 45 |

Actions : 2 913 labels conservés par défaut, 1 298 conservations explicites,
246 corrections, 92 passages en négatif, 2 603 exclusions.
Treize images multi-boîtes sont conservées (une train et douze test).
L'audit structurel est valide ; aucune séquence ni doublon exact ne traverse
les splits. Les proportions ont évolué avec le tri sans réaffectation des séquences.

Cette version est triée par l'utilisateur avec conservation par défaut, pas
une vérité terrain entièrement réannotée : des erreurs peuvent rester dans
les labels d'origine, y compris en test. Le rapport complet et les instantanés
de décisions sont dans `data/processed/v002`; le résumé reproductible est dans
`models/reports/infactory_v002_summary.json`. Les sources brutes et v001 restent
disponibles. Le prochain jalon est le premier entraînement de référence.
