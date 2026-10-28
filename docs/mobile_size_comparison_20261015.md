# Comparaison Nano 1280 et Small 960 — 15 octobre 2026

Deux entraînements soumis à Hugging Face Jobs à la demande de l’utilisateur. Chaque job utilise une A10G à environ 1 USD/h, timeout 90 min : coût GPU total plafonné à environ 3 USD. Les dépôts de modèles sont privés. Aucun entraînement dupliqué ni nouvelle soumission automatique.

## Protocole commun

- Export figé mobile_v004, 2 806 images train / 168 val, sans TV ; même révision dataset 77c48dac63cd314bd3d0a54b9d2caf2be0ed4641 et même archive SHA-256 a668ba60c58a64afe24c68e0227e9cfa11e4d411ef82019f85dd2318378f293d.
- Initialisation COCO de chaque architecture, transfert complet du classifieur sports ball (ligne 32 vers 0), sans repartir des poids affinés du v004.
- 30 époques maximum, patience 8, batch 8, seed 42, AdamW, lr0 0,001 et augmentations inchangées. Checkpoints privés toutes les 2 époques.
- Référence : Total v004 Nano 960, réévalué à la résolution propre à chaque essai (1280 ou 960), sur la même validation. Aucun test réservé n’est utilisé pour sélectionner le checkpoint.
- Les clips iPhone fournis le 15 octobre restent hors de cet export figé ; les comparer ensuite sans les réinjecter dans cet entraînement.

## Adaptation et vérification du pipeline

Le lanceur et le worker prennent maintenant le checkpoint base_model configuré, au lieu d’imposer Nano. Small conserve ses 128 canaux de classifieur, Nano ses 80, dans les deux branches. Les 25 tests passent : identité des poids transférés, prédictions brutes et boîtes décodées, loss finie et gradients, sauvegarde/rechargement, trainer, poursuite et protection contre les soumissions payantes en double. Les vérifications locales complètes avant upload passent pour les deux architectures ; les hashes des images/labels figés sont vérifiés. Le worker reproduit les contrôles sur GPU.

L’app conserve le modèle Total actuel. La conversion Core ML et les performances réelles sur iPhone restent à vérifier après entraînement ; un temps GPU A10G ne mesure pas le temps iPhone.

## Jobs et commandes

- yolo26n.pt à 1280 px : https://huggingface.co/jobs/Totorotibo84/6ad14bbc5ccb47ba0f7ad330 ; reçu `models/reports/yolo26n_mobile_field_v004_1280_20261015_job.json`.

```sh
.venv/bin/python scripts/mobile_training_hf.py status --config configs/mobile_nano_field_all_balls_1280.json
.venv/bin/python scripts/mobile_training_hf.py collect --config configs/mobile_nano_field_all_balls_1280.json
```

- yolo26s.pt à 960 px : https://huggingface.co/jobs/Totorotibo84/6ad14bd6c662881452255e2c ; reçu `models/reports/yolo26s_mobile_field_v004_960_20261015_job.json`.

```sh
.venv/bin/python scripts/mobile_training_hf.py status --config configs/mobile_small_field_all_balls_960.json
.venv/bin/python scripts/mobile_training_hf.py collect --config configs/mobile_small_field_all_balls_960.json
```


## Résultats collectés le 16 octobre 2026

Même validation : 168 images, 134 ballons annotés, seuil de confiance 25 %, IoU de correspondance 50 %.

| Modèle | TP | FP | FN | Précision | Rappel | F1 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo26n.pt 960 | 95 | 23 | 39 | 80.5 % | 70.9 % | 75.4 % | 45.8 % |
| yolo26n.pt 1280 | 92 | 31 | 42 | 74.8 % | 68.7 % | 71.6 % | 45.9 % |
| yolo26s.pt 960 | 101 | 21 | 33 | 82.8 % | 75.4 % | 78.9 % | 49.4 % |

Small 960 présente le meilleur bilan global : six TP supplémentaires, six FN de moins et deux FP de moins que Total actuel. Sur les images de validation sans ballon, ses FP passent cependant de 4 à 8. Nano 1280 fait moins bien au seuil fixé ; augmenter la résolution n’a pas suffi dans cet essai. Les FP sont définis par les annotations existantes, dont la couverture des ballons secondaires reste une limite. La validation ne mesure pas spécifiquement les petits ballons des deux nouveaux clips nocturnes.

Les deux jobs sont COMPLETED. Nano1280 : 15 époques, meilleur mAP50–95 à l’époque 7, 19,1 minutes d’entraînement. Small960 : 26 époques, meilleur mAP50–95 à l’époque 18, 25,1 minutes d’entraînement. Les deux ont atteint le critère d’arrêt après huit époques sans amélioration. Les meilleurs poids et les courbes sont récupérés localement ; leurs SHA-256 figurent dans le rapport JSON.

Le Nano Total actuel réévalué directement en 1280 retrouve 103 ballons mais produit 62 FP : ce contrôle confirme qu’un agrandissement peut aussi faire apparaître davantage de fausses propositions. Le Nano réentraîné en 1280 ramène les FP à 31 mais ne dépasse pas le bilan du Nano960 existant.

Les temps rapportés par les jobs sont des mesures A10G, pas des mesures iPhone. Aucun export Core ML, remplacement de modèle ou installation iPhone n’a été effectué lors de cette récupération des résultats.
