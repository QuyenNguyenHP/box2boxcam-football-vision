# PitchLens — données et entraînement

Pipeline d’annotation, de préparation des datasets, d’entraînement YOLO et
d’évaluation pour filmer automatiquement le football avec un iPhone sur Osmo.
L’application, le suivi Swift et l’atelier de rejeu sont dans
[pitchlens-ios](https://github.com/Thibault-GILLARD/pitchlens-ios).

## État actuel

Le modèle ballon utilisé par l’app est **Total Small** : YOLO26 Small field v004,
entrée de 960 px, checkpoint
`models/checkpoints/baseline_runs/yolo26s_mobile_field_v004_960_20261015/artifacts/best.pt`.
Les variantes Nano 960 et Nano 1280 restent des références de comparaison.
L’app combine le ballon avec les personnes et les poses ; ce dépôt contient la
chaîne qui prépare et évalue les détecteurs.

La rencontre iPhone du 14 octobre, échantillonnée en 568 images, reste réservée
hors entraînement. La revue humaine est en cours : les propositions de Luna ne
sont pas des vérités terrain. Voir [le protocole terrain](docs/field_validation_20261021.md)
et [la comparaison des tailles de modèles](docs/mobile_size_comparison_20261015.md).

## Installation

Python 3.11 ou plus récent :

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,training,annotation,notebook]'
cp .env.example .env
```

Renseigner les accès nécessaires dans le fichier `.env` local. Les scripts de
soumission d’annotations ou d’entraînement déclenchent des services payants ;
les tests ci-dessous n’en soumettent pas.

## Où travailler

| Besoin | Point d’entrée |
| --- | --- |
| Sources et recettes | `configs/` et [audit des données](docs/dataset_assessment.md) |
| Préparation et contrôle des données | `src/pitchlens/data/` et `notebooks/01` à `03` |
| Annotation Luna et revue | `notebooks/07_luna_batch_annotation.ipynb`, `configs/label_studio_field_validation.xml` |
| Entraînement mobile | `scripts/mobile_training_hf.py`, worker `scripts/train_mobile_hf.py`, `notebooks/09_mobile_training.ipynb` |
| Configuration du dataset courant | [mobile field v004](docs/mobile_field_v004_20261014.md) |
| Évaluation | `src/pitchlens/training/`, `scripts/evaluate_mobile_personal.py`, `models/reports/` |
| Pistes de recherche | [vision et cadrage](docs/vision_etat_de_lart_20261105.md) |

Dans Label Studio, **Submit** confirme les boîtes sans deuxième case obligatoire.
Si l’image ne peut pas être annotée avec certitude, cocher **Image inexploitable**
avant Submit. Elle est exclue, sans devenir une image négative. Tous les ballons
visibles, y compris ceux de réserve, comptent ; le rôle actif reste facultatif.

## Vérifier

```sh
python -m unittest discover -s tests
python -m ruff check src scripts tests
```

Les tests utilisent des données temporaires et des clients simulés. Les
notebooks sont versionnés sans sorties d’exécution.

## Conservation des fichiers

GitHub contient le code, les recettes, notebooks et rapports légers. Les vidéos,
images de dataset, environnements et checkpoints restent hors Git ; les artifacts
lourds sont conservés séparément, notamment sur Hugging Face. Les manifestes et
empreintes permettent de retrouver les versions utilisées. Une nouvelle rencontre
de validation doit rester absente de l’entraînement.

Les anciens essais, recettes et résultats détaillés ont été déplacés dans
[le journal de développement](docs/DEVELOPMENT_LOG.md).
