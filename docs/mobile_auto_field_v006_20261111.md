# YOLO26s mobile v006 — candidats jugés automatiquement, revus par clics

Entraînement du 11 novembre 2026, job HF `6af43afc9737d9a9e092c153`, 30 époques, même recette
que Small v004 (`configs/mobile_small_auto_field_v006_960.json`). Seul changement : le train
ajoute `auto_field_judged_v004` (477 images, 360 boîtes, 352 masques gris).

## Source ajoutée

`auto_field_judged_v004` = lot 1 (208 images) + lot 2 (300 images) jugés par Claude Sonnet,
puis revue humaine par clics des vignettes « ballon » et « incertain » : 21 « ballon » rejetés
(passés en masque), 63 « incertain » promus en « ballon ». Les « pas un ballon » sont restés
tels que jugés (section parcourue par l'utilisateur, aucune correction).

## Test réservé agrandi

Le test du 14 octobre passe de 250 à 396 images (219 ballons, dont 149 de 16 px ou moins) :
146 images en attente revues par clics le 11 novembre (132 autres exclues). Les boîtes des
nouvelles images sont celles des candidats (Luna ou détecteur v004), pas des tracés manuels ;
23 des 125 nouveaux ballons n'étaient proposés que par le détecteur v004.

## Résultats (TP / FP, IoU ≥ 0,5, imgsz 960)

| Seuil | Small v004 | v005 | v006 |
|---|---:|---:|---:|
| 0,25 | 95 / 203 | 97 / 136 | 100 / 103 |
| 0,35 | 93 / 113 | 93 / 101 | 93 / 60 |
| 0,45 | 89 / 64 | 88 / 61 | 86 / 30 |
| 0,55 | 81 / 23 | 82 / 16 | 81 / 3 |

TP maximal pour un budget de FP : ≤ 25 → 81 / 83 / 84 ; ≤ 60 → 88 / 87 / 93 ; ≤ 100 → 91 / 92 / 99 ;
≤ 200 → 95 / 102 / 109. Bootstrap par image, v006 − v004 : à 0,25 dTP [−4, +5, +14], dFP [−132, −99, −69] ;
à 0,45 dTP [−11, −3, +4], dFP [−50, −34, −19].

Ancienne validation (168 images, seuil 0,25) : v004 101 TP / 21 FP / 33 FN, mAP50-95 49,4 % ;
v006 96 / 19 / 38, 48,3 % (FP sur négatives 8 → 4).

Conclusion : v006 fait nettement moins de fausses boîtes sur le match réservé à rappel égal ;
le rappel ne progresse pas de façon démontrée. Pas d'export Core ML à ce stade.
