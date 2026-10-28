# Nano 960 : les anciennes images TV aident-elles ?

Cette seconde expérience ajoute le **train de v002**, sans reprendre ses
anciennes validations ni ses tests. Les images déjà exclues restent exclues.
Les annotations exportées après correction sont utilisées, pas les anciennes
propositions YOLO. La qualité est hétérogène : cet essai ne suppose pas que
toutes les images TV sont mauvaises, ni que leur ajout sera bénéfique.

| Données | Récent seul (terminé) | Récent + TV (nouvel essai) |
|---|---:|---:|
| Train récent | 1 547 | 1 547 |
| Train TV v002 | 0 | 3 048 |
| Total train | 1 547 | 4 595 |
| Train avec ballon | 1 390 | 3 503 |
| Train sans ballon | 157 | 1 092 |
| Validation, strictement identique | 168 | 168 |
| Images de test utilisées pour entraîner | 0 | 0 |

La comparaison se fait avec **YOLO26n à 960 px**, la même initialisation COCO,
le même batch de 8, AdamW, seed 42, les mêmes augmentations, 60 époques maximum
et patience 12. Le modèle récent n'est **pas** le point de départ du nouvel
entraînement : son `best.pt` sert de référence sur la validation commune.
Il reste intact et les résultats du nouvel essai ont leur propre dossier.

Attention à l'interprétation : nous changeons simultanément la quantité,
le domaine visuel, la qualité des annotations et la proportion de négatifs
(10,1 % → 23,8 %). Une époque contient aussi davantage de mises à jour.
Ce n'est donc pas une mesure isolée de l'effet de la « qualité moyenne ».
Les métriques diront surtout si **cet ajout précis** aide notre modèle.

Le checkpoint est sélectionné sur la validation, jamais sur les 292 images
personnelles. Ce benchmark personnel a déjà été regardé ; une nouvelle
comparaison dessus sera exploratoire, pas un nouveau test aveugle. Le clip
court de test et le clip long de train partagent le match du 30 septembre.
Un autre match entièrement réservé serait nécessaire pour une mesure
indépendante de la généralisation.

## Fichiers et suivi

- Configuration : `configs/mobile_nano_tv_960.json`.
- Export dérivé : `data/processed/mobile_tv_v001`, avec manifestes et audit.
  Les JPEG sont des liens physiques, sans deuxième copie permanente.
- Présentation : section 10 du notebook existant `09_mobile_training.ipynb`.
- Premier modèle : `models/checkpoints/baseline_runs/yolo26n_mobile_v001_960_20261010/artifacts/best.pt`.
- Nouvel identifiant : `yolo26n_mobile_tv_v001_960_20261010`.

Le job privé HF est limité à deux heures sur A10G small. Au tarif constaté
de 1 $/h, cela donne un plafond de calcul d'environ 2 $, et non une prévision
du coût réellement facturé. Le délai est doublé par rapport au premier essai
pour laisser le même nombre maximal d'époques avec trois fois plus de train.

Job soumis le 10 octobre :
[yolo26n_mobile_tv_v001_960_20261010](https://huggingface.co/jobs/Totorotibo84/6aca667f095c57808931454b).
Le job est terminé : 59 époques, 4 574,8 secondes d'entraînement
(environ 76 minutes). Le checkpoint final sélectionné sur la validation a été
récupéré dans `models/checkpoints/baseline_runs/yolo26n_mobile_tv_v001_960_20261010/artifacts/best.pt`.

## Résultats finaux sur la validation commune

Les métriques au seuil fixe utilisent confiance ≥ 0,25 et IoU ≥ 0,50.

| Mesure | Récent seul | Récent + TV |
|---|---:|---:|
| mAP50 | 75,6 % | 75,0 % |
| mAP50–95 | 42,5 % | 46,1 % |
| Précision au seuil fixe | 73,0 % | 88,8 % |
| Rappel au seuil fixe | 68,7 % | 64,9 % |
| F1 au seuil fixe | 70,8 % | 75,0 % |
| Bonnes détections | 92 | 87 |
| Fausses détections | 34 | 11 |
| Ballons ratés | 42 | 47 |
| Fausses détections sur images négatives | 13 | 3 |

L'ajout améliore la localisation globale et réduit nettement les faux
positifs, avec un compromis défavorable sur le rappel au seuil choisi :
23 fausses détections de moins mais 5 ballons ratés de plus. Ce n'est donc pas
une amélioration uniforme. Le benchmark personnel et les gros plans n'ont
pas encore été réévalués avec ce nouveau checkpoint ; les résultats ci-dessus
ne prouvent pas une amélioration sur ces situations.

Ces commandes ne lancent pas de nouveau job payant :

```bash
python scripts/mobile_training_hf.py status --config configs/mobile_nano_tv_960.json
python scripts/mobile_training_hf.py collect --config configs/mobile_nano_tv_960.json
```

`collect` refuse de présenter un entraînement incomplet comme un résultat final.
Le job a été collecté ; la section 10 affiche désormais la comparaison
sur les mêmes 168 images de validation, les matrices de confusion et les
courbes d'apprentissage du nouvel essai.
