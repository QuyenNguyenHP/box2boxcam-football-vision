# Réutiliser réellement la classe COCO sports ball

Nouvel essai : `yolo26n_mobile_sports_ball_v001_960_20261010`.
Configuration : `configs/mobile_nano_sports_ball_960.json`.

## Ce que nous avons vérifié

Dans le modèle COCO, `sports ball` est la classe **32**. Dans notre modèle
mono-classe elle devient **0**, avec le même nom. Le numéro 0 ne correspond
donc pas à « person » : cette numérotation était celle de la liste COCO.

Avec Ultralytics **8.4.153**, un simple changement de nom ne suffit pas
pour YOLO26n : le constructeur réduit les canaux internes du classifieur de
80 à 64 lorsque nc passe de 80 à 1. Le remappage par noms conserve alors les
six biais, mais pas les six lignes de poids de sortie dont les dimensions
diffèrent. C'est confirmé par le code local et la
[documentation du remappage](https://docs.ultralytics.com/reference/nn/tasks/).

Pour conserver le classifieur appris, `sports_ball_transfer.py` :

1. Copie le graphe COCO préentraîné sans modifier l'original.
2. Conserve les caractéristiques visuelles et la localisation.
3. Dans les trois échelles de **chacune des deux branches** de détection,
   garde uniquement les poids et biais de la sortie 32.
4. Renumérote cette sortie en 0, nommée `sports ball` ; nc devient 1.
5. Garde les 80 canaux internes, au lieu de reconstruire une tête à 64.

Un trainer adapte uniquement la construction du modèle, selon le mécanisme
de [personnalisation Ultralytics](https://docs.ultralytics.com/usage/engine/).
La loss, l'optimiseur, l'EMA, la validation et les sauvegardes restent ceux
d'Ultralytics. Le checkpoint contient un `DetectionModel` standard, pas une
nouvelle classe de modèle dépendant de notre helper pour l'inférence.

## Preuves avant entraînement

- Les 12 tenseurs de sortie correspondent exactement à la ligne COCO 32.
- Les 696 autres tenseurs sont conservés et la source reste inchangée.
- Les deux branches sont testées : logits ballon identiques à l'arrondi
  flottant près, boîtes brutes et décodées identiques.
- La sauvegarde/relecture conserve le graphe et les prédictions.
- La loss de détection et les gradients des deux branches sont finis.
- Le trainer conserve réellement la largeur 80, sans reconstruction cachée.

Ces contrôles sont dans `tests/test_sports_ball_transfer.py`. Le lanceur
répète le contrôle avant soumission, et le worker GPU avant l'optimisation.
Les résultats sont enregistrés dans le job JSON (`transfer_preflight`) et
dans le rapport d'entraînement (`sports_ball_transfer`). Le contrôle des
scores porte sur les prédictions **avant le top-k** : supprimer les autres
classes peut modifier cette sélection, même si les scores bruts sont conservés.

## Données et comparaison

**Aucune image Infactory/TV de v002 ajoutée**. Nous réutilisons la même révision privée du dataset
`mobile_v001` : 1 547 train, dont 1 390 avec ballon et 157 négatifs ; 168 val,
dont 134 avec ballon et 34 négatifs. Les annotations restent numérotées 0.
Images et labels sont vérifiés par SHA-256 ; aucun nouvel upload d'images ni
nouvelle copie locale n'est nécessaire. Seul le YAML du worker prend le nom
`sports ball`. Aucun test n'est utilisé pour entraîner ou sélectionner le poids.

Les autres réglages restent ceux du premier essai récent : YOLO26n, 960 px,
batch 8, AdamW, lr 0,001, seed 42, 60 époques maximum, patience 12, mêmes
augmentations. L'initialisation repart du COCO original, **pas** du best.pt
récent, qui sert seulement de référence sur les mêmes 168 images.

Le modèle a 2 533 886 paramètres, contre 2 504 190 dans l'ancien mono-classe
(+1,19 %), car la largeur interne préentraînée est conservée. Nous testons
donc le transfert complet du classifieur, **pas uniquement un renommage à
architecture strictement identique**. Les poids restent libres d'évoluer
pendant l'entraînement : cela ne garantit pas que toutes les capacités de
gros plan seront conservées. Il faudra des images dédiées pour le mesurer.

Le job privé utilise A10G small, avec une limite d'une heure. Au tarif constaté
de 1 $/h, le plafond de calcul est environ 1 $. Les anciens poids sont préservés.

Job soumis le 10 octobre :
[yolo26n_mobile_sports_ball_v001_960_20261010](https://huggingface.co/jobs/Totorotibo84/6acaafa2095c578089317ff5).
Le démarrage sur GPU est confirmé : le log vérifie le transfert complet
32 → 0 et 2 533 886 paramètres avant de commencer les 60 époques maximum.
Les résultats finaux ne sont pas encore disponibles à ce stade.
Avant soumission, les 77 tests du projet passaient. Un vrai mini-entraînement
CPU a également été vérifié : 4 images positives de train, 2 de validation,
1 époque à 128 px, loss finie et checkpoint rechargeable avec la largeur 80.
Ce smoke test temporaire ne sert pas de mesure de performance.

```bash
python scripts/mobile_training_hf.py status --config configs/mobile_nano_sports_ball_960.json
python scripts/mobile_training_hf.py collect --config configs/mobile_nano_sports_ball_960.json
```

La section 11 du notebook existant `09_mobile_training.ipynb` présentera la
comparaison finale, les matrices et les courbes après collecte.

Attention pour une future reprise : `YOLO(best.pt).train()` avec le trainer
standard peut reconstruire la tête à largeur 64. Utiliser à nouveau
`SportsBallTransferTrainer` pour préserver les canaux appris. L'inférence
ordinaire avec `YOLO(best.pt)` n'a pas cette limitation.
