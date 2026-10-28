# YOLO26n mobile v004 — match iPhone annoté

Le nouveau lot `mobile_v004` reprend les 2 826 entrées figées de `mobile_v003`
et ajoute 148 images du match iPhone validées dans Label Studio : 141 positives,
7 négatives, 148 boîtes au total, dont 6 images contenant plusieurs ballons.
Les 1 095 images du match encore non soumises ou en brouillon ne sont pas utilisées.
Les corrections soumises des anciens projets ont été comparées aux exports :
elles correspondent aux labels déjà présents dans la référence.

| Source | Entraînement | Validation |
|---|---:|---:|
| Compilation Luna de base | 739 | 168 |
| Lou Mistrau personnel | 808 | 0 |
| Shorts déjà revus | 603 | 0 |
| Ballons proches | 445 | 0 |
| Faux positifs : vidéos sans ballon | 63 | 0 |
| Match iPhone : annotations humaines | 148 | 0 |
| **Total** | **2 806** | **168** |

Le train contient 2 415 images positives et 391 négatives. Les données télé
Infactory sont exclues. Les 168 images et labels de validation conservent exactement
leurs empreintes précédentes ; les images réservées au test sont absentes du lot GPU.
Les anciennes références, les vidéos et les annotations de l'outil de revue restent intactes.

## Point de départ et recette

L'expérience configurée est un nouveau départ depuis **YOLO26n préentraîné COCO**,
sur toutes les données réunies. Ce n'est pas une initialisation aléatoire.
Le modèle v003 sert de référence de comparaison sur la validation, pas de poids initiaux.
La tête `sports ball` conserve les caractéristiques du classifieur COCO et sa ligne
de sortie grâce au transfert déjà vérifié dans les expériences précédentes.

Repartir des poids v003 et réentraîner sur ce même lot serait aussi un entraînement
sur toutes les données, mais avec un autre point de départ et une recette de continuation.
Entraîner sur les 148 nouvelles images seules serait encore un troisième protocole ;
il n'est pas utilisé ici.

Résolution 960 px, batch 8, AdamW, lr initial 0,001, seed 42,
maximum 30 époques, patience 8, GPU A10G, limite totale de 90 minutes.
Les checkpoints sont publiés en privé toutes les deux époques. La sélection du
meilleur checkpoint utilise uniquement la validation, jamais le test.

Le nombre d'époques, le point de départ et l'ajout des images changent ensemble :
la comparaison à v003 mesure cette nouvelle recette complète, pas l'effet isolé
des 148 images. La validation existante couvre peu les gros plans et ne contient
pas cette nouvelle session de match. Elle ne suffit donc pas à mesurer la
généralisation sur un autre match nocturne. Le diagnostic des 63 faux positifs
est effectué sur des images d'entraînement, et reste décrit comme tel.

La nouvelle session annote tous les ballons visibles, y compris ceux immobiles
ou au bord du terrain. Les anciens lots avaient une politique « ballon en jeu » ;
leurs ballons secondaires éventuellement omis restent un point à harmoniser
dans les prochaines revues.

## Fichiers et commandes

- Configuration : `configs/mobile_nano_field_all_balls_960.json`.
- Export humain : `data/processed/field_all_balls_reviewed_v001`.
- Lot entraînement/validation : `data/processed/mobile_v004`.
- Reçu GPU : `models/reports/yolo26n_mobile_field_v004_960_20261014_job.json`.
- Résultats collectés : `models/checkpoints/baseline_runs/yolo26n_mobile_field_v004_960_20261014`.

```sh
.venv/bin/python scripts/mobile_training_hf.py status --config configs/mobile_nano_field_all_balls_960.json
.venv/bin/python scripts/mobile_training_hf.py collect --config configs/mobile_nano_field_all_balls_960.json
```

Les dépôts Hugging Face du dataset et du modèle sont privés. Le lancement refuse
un job déjà enregistré ; les anciens entraînements ne sont jamais remplacés.
Le job [6acf98865ccb47ba0f79c275](https://huggingface.co/jobs/Totorotibo84/6acf98865ccb47ba0f79c275)
est lancé, avec la révision dataset `77c48dac63cd314bd3d0a54b9d2caf2be0ed4641`
et la révision de code `1c68ef4700f65500b8b5ed35c31f360657a7a069`.
La récupération finale est lancée par `scripts/collect_mobile_job_when_done.py` ;
elle ne soumet aucun entraînement et refuse de présenter un checkpoint de progression
comme résultat terminé. Son journal est dans
`models/reports/yolo26n_mobile_field_v004_960_20261014_collector.log`.
Le gel de la nouvelle revue est implémenté dans
`src/pitchlens/data/build_reviewed_field.py` et vérifié par les tests de sélection
humaine, de négatifs explicites, de plusieurs boîtes et de brouillons exclus.

## Résultats terminés et récupérés

Le job est terminé et les artefacts sont récupérés localement, à la révision
`1ddd1e9af4adabd2015c4948f95f1e234319e425`. Les 30 époques ont été effectuées
en 25,5 minutes de temps d'entraînement ; le meilleur mAP50-95 de validation
dans la courbe correspond à l'époque 28.

| Même validation de 168 images | V003 | V004 |
|---|---:|---:|
| mAP50 | 75,1 % | 76,2 % |
| mAP50-95 | 43,6 % | 45,8 % |
| Ballons détectés (TP) | 90 | 95 |
| Ballons manqués (FN) | 44 | 39 |
| Prédictions non associées aux annotations (FP) | 15 | 23 |
| Précision | 85,7 % | 80,5 % |
| Rappel | 67,2 % | 70,9 % |
| F1 | 75,3 % | 75,4 % |

Les comptes TP/FP/FN et précision/rappel/F1 utilisent le même seuil 0,25 et
le même IoU d'association 0,50. V004 récupère 5 ballons supplémentaires et
produit 8 prédictions supplémentaires non associées à une annotation. Le F1
change de seulement 0,08 point : pas de gain global net à ce point de fonctionnement.
Le gain mAP50-95 suggère une meilleure détection/localisation sur cette validation,
mais ne prouve pas une amélioration sur un autre match ni sur les ballons de réserve.
Le diagnostic des 63 images négatives d'entraînement reste à zéro fausse boîte
pour les deux modèles ; il ne constitue pas un test indépendant.

Les annotations historiques de validation ciblent le ballon en jeu ; la nouvelle
session annote tous les ballons visibles. Un FP mesuré est donc une prédiction
non associée à ces annotations, pas automatiquement la preuve que l'objet visible
n'est pas un vrai ballon. Les boîtes supplémentaires demandent une inspection.

Modèle final :
`models/checkpoints/baseline_runs/yolo26n_mobile_field_v004_960_20261014/artifacts/best.pt`.
SHA-256 : `cd3ad62956b77987d830414231fcaa9246350cb8caaaace8c3923b6213d38a71`.
Le v003 est conservé pour comparaison. Le test réservé n'a pas servi à la sélection,
et les 148 images du nouveau match étaient dans le train, pas dans cette validation.
