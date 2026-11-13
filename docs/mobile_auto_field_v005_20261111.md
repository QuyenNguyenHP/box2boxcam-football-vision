# YOLO26s mobile v005 — candidats jugés automatiquement

Nouvelle source `auto_field_judged_v001` : 208 images iPhone non revues (match du
12 octobre et clips Lou Mistrau du 30 septembre et du 7 octobre), absentes de
`mobile_v004`, de la validation, du clip de test réservé et de la session réservée
du 14 octobre. Leurs 999 candidats (détecteur plein cadre, détecteur par tuiles,
Luna) ont été jugés sur des recadrages zoomés par un modèle de vision :
179 `ball`, 650 `not_ball`, 170 `unsure`.

**Aucun humain n'a vérifié ces labels.** Le champ `label_source` vaut
`auto_judged_claude_sonnet` dans les manifestes et le README de `mobile_v005`
le signale.

## Politique de label

- Image écartée si un de ses candidats dépasse 64 px de côté (le juge ne le voyait pas en entier).
- `ball` : une boîte YOLO, classe 0, boîte du candidat bornée à l'image.
- `not_ball` : aucun label, l'objet reste du fond (négatif difficile).
- `unsure` : ni positif ni négatif. Rectangle gris (114, 114, 114) sur la boîte
  agrandie de 50 % (au moins 24 px de côté), dans une copie JPEG qualité 95.
  Les images sans `unsure` restent des liens physiques vers le JPEG d'origine.
- Image écartée si une boîte `ball` chevauche un masque.
- Tout va dans le train. Le manifeste garde, par image, le chemin et le SHA-256
  source, le match, le clip, le masquage et, par candidat, verdict, sources et confiances.

## Calibration et limites

À l'aveugle contre des labels humains, sur des candidats du détecteur, le juge a
raison dans 98,7 % des cas quand il est confiant : 1 `not_ball` faux sur 272,
3 `ball` faux sur 45. Les 11 boîtes `ball` proposées par Luna seule sortent de ce
périmètre de calibration.

Un ballon proposé ni par le détecteur ni par Luna reste non annoté : il devient
du fond.

Contrôle visuel rapide des 84 `not_ball` gardés avec une confiance détecteur ≥ 0,6 :
le candidat 150 (`2026-09-30_lou_mistrau_19h38m04_frame_00002412`, confiance 0,65)
est un vrai ballon flou au pied du joueur n° 38, entraîné ici comme fond.
Quelques autres (931, 451, 248) restent ambigus. Ce taux reste compatible avec la
calibration, mais ces erreurs touchent justement les négatifs difficiles.

## Chiffres

| | Valeur |
|---|---:|
| Images d'entrée | 208 |
| Écartées : candidat > 64 px | 10 |
| Écartées : ballon sous un masque | 2 |
| Gardées | 196 |
| Positives / négatives | 119 / 77 |
| Boîtes | 175 |
| Images masquées / masques | 106 / 148 |
| `not_ball` gardés avec confiance détecteur ≥ 0,25 | 310 |

Par match : session du 12 octobre 138 images (87 positives, 125 boîtes),
Lou Mistrau 7 octobre 40 (23 positives, 36 boîtes), Lou Mistrau 30 septembre 18
(9 positives, 14 boîtes).

`mobile_v005` reprend `mobile_v004` à l'identique et ajoute la source en train :
3 002 images de train (2 534 positives, 468 négatives) et les mêmes 168 images de
validation. Les empreintes image/label de la validation sont identiques à v004.

## Entraînement préparé

Configuration `configs/mobile_small_auto_field_960.json` : même recette que
Small v004 (YOLO26s COCO, transfert `sports ball`, 960 px, 30 époques, patience 8,
batch 8, AdamW lr0 0,001, seed 42, A10G, 90 min, dépôts privés). La référence
évaluée sur la même validation est le Small v004 actuel. Seul le train change, mais
ces images sont proches dans le temps d'images déjà présentes dans le train.

```sh
.venv/bin/python -m pitchlens.data.build_auto_judged_field <dossier_du_juge>
.venv/bin/python scripts/mobile_training_hf.py prepare --config configs/mobile_small_auto_field_960.json
.venv/bin/python scripts/mobile_training_hf.py launch --config configs/mobile_small_auto_field_960.json
```

`prepare` construit seulement `mobile_v005` en local. `launch` crée les dépôts
privés, envoie l'archive et le code, puis soumet aussitôt le job GPU payant
(environ 1 USD/h, plafond 90 min). Lancé le 11 novembre ; voir les résultats ci-dessous.

## Résultats — 11 novembre 2026

Job [6af3c9a99737d9a9e0925575](https://huggingface.co/jobs/Totorotibo84/6af3c9a99737d9a9e0925575) terminé : 30 époques, 30,9 minutes d'entraînement. Checkpoint récupéré dans
`models/checkpoints/baseline_runs/yolo26s_mobile_auto_field_v005_960_20261111/artifacts/best.pt`.

Validation inchangée (168 images, 134 ballons, seuil 0,25, IoU 0,50) :

| Modèle | TP | FP | FN | FP sur images négatives | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small v004 | 101 | 21 | 33 | 8 | 49,4 % |
| Small v005 | 100 | 19 | 34 | 3 | 52,3 % |

Match réservé du 14 octobre, 250 images revues de `recording_01` (94 ballons, 60 petits), évaluation
seule, PyTorch image entière à 960 px :

| Modèle | Seuil 0,25 TP / FP | Seuil 0,45 TP / FP | Seuil 0,55 TP / FP | FP pour 30 TP | FP pour 37 TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small v004 | 37 / 133 | 34 / 45 | 29 / 17 | 19 | 81 |
| Small v005 | 34 / 92 | 30 / 48 | 28 / 13 | 48 | 180 |

Au seuil 0,25, v005 produit moins de fausses boîtes (intervalle bootstrap par image à 95 % : −66 à −16)
mais retrouve aussi moins de ballons (−8 à +2). À rappel égal, il n'est pas meilleur : ses scores sont
décalés vers le bas, le classement des détections ne s'améliore pas. **v005 n'est pas retenu pour
l'app ; Small v004 reste le modèle courant.** Aucun export Core ML n'a été fait.

Lecture : 196 images ajoutées à 2 806 est une dose faible, proches d'images déjà présentes, et les
fausses détections dominantes du match réservé (projecteur, tache sur la pelouse, chaussures de ce
stade) ne sont pas dans ces images. Le script d'évaluation est conservé hors dépôt ; un seul clip,
un seul stade.
