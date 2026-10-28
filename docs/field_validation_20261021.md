# Validation terrain iPhone — session du 14 octobre 2026

Cette session est réservée à la **validation de développement**, hors entraînement.
Elle a déjà été regardée et comparée qualitativement : ce n'est pas un test final
aveugle. Les deux enregistrements appartiennent à **une seule rencontre** ; ils ne
constituent pas deux matchs indépendants. La session d'entraînement iPhone est
celle du 12 octobre, identifiée séparément.

## Échantillon figé

| Source originale | Identifiant | Durée | Images à revoir |
| --- | --- | --- | --- |
| `match 2.mov` | `recording_01` | 157,29 s | 315 |
| `match.mov` | `recording_02` | 126,39 s | 253 |

Les **568 images** couvrent uniformément les deux clips : une image toutes les
15 images décodées, soit environ 2 Hz. Aucun filtrage par confiance ou choix de
passages favorables. Chaque tâche fournit l'image centrale et le contexte à
environ −0,2/+0,2 s. Les fichiers restent en 1920 × 1080, JPEG qualité 95 ; les
SHA-256 et les temps de présentation décodés sont conservés.

Cette référence à 2 Hz convient à une première comparaison de détection. Elle ne
permet pas de mesurer précisément les pertes de suivi inférieures à 500 ms.
Le suivi et le cadrage à 30 Hz demanderont une annotation temporelle plus dense.

## Revue humaine

Projet local : http://127.0.0.1:8080/projects/7/data

Luna (`gpt-6-luna`, effort `max`) propose des boîtes à partir des trois images.
Les modèles Nano/Small évalués ne fournissent pas les préannotations. Cela
n'élimine pas les erreurs de Luna : **aucune proposition n'est une vérité terrain**.

Depuis le 23 octobre, **Submit est le seul clic de confirmation nécessaire**.
La grande image s'affiche seule ; le contexte et les options sont repliés.
Pour chaque image :

1. Corriger, ajouter ou supprimer les boîtes pour couvrir **tous les vrais ballons
   visibles**, y compris les ballons de réserve. Ne pas deviner les positions
   cachées. Une boîte serre la partie visible du ballon.
2. Cliquer **Submit**. Si les propositions sont déjà correctes, aucune autre
   action n'est nécessaire. La case `Vérifié` n'est plus obligatoire.

Si l'image entière ne peut être annotée de manière fiable, ouvrir les options
facultatives et choisir `Image inexploitable`. Elle sera exclue du dénominateur,
jamais convertie en image négative.

L'identification du ballon actif est maintenant **facultative** : dans les
options, attribuer les rôles `Actif`, `Secondaire` ou `Rôle indéterminé`, choisir
l'état de visibilité et cocher `Identité vérifiée` uniquement après cette revue.
`Actif visible` exige exactement une boîte active. Sans cette confirmation
supplémentaire, l'export conserve les boîtes vérifiées mais classe l'identité
active comme indéterminée, avec `active_identity_reviewed: false`.
Les propositions de rôle restent conservées dans les annotations brutes, sans
être transformées en vérité terrain de suivi.

Les images sans ballon visible doivent aussi être explicitement vérifiées.
Une réponse API manquante, un brouillon ou une tâche ignorée reste en attente.
Le protocole `submit_boxes_v2` accepte uniquement une annotation réellement
soumise ; aucune tâche n'est validée automatiquement. Les anciennes annotations
sans ce protocole conservent leur contrôle explicite `Vérifié`.

## Commandes et reprise

Depuis la racine du dépôt, le processus ponctuel
`scripts/run_field_validation_pipeline.py` attend les lots déjà soumis, collecte
les réponses et synchronise les propositions dans ce projet. Les IDs de lots et
fichiers sont persistés ; ne pas lancer un second processus pour les remplacer.
Un échec ne déclenche pas automatiquement de nouvelle soumission payante.

```sh
# État de la revue : ne lance ni Luna ni entraînement.
.venv/bin/python scripts/review_field_validation.py status

# Reprendre uniquement la synchronisation de réponses déjà collectées.
# Ne remplace aucune annotation ni aucun brouillon humain existant.
.venv/bin/python scripts/review_field_validation.py sync

# Après la revue complète ; refuse tant qu'une tâche manque ou est incohérente.
.venv/bin/python scripts/review_field_validation.py freeze
```

L'export immuable sera créé dans `data/processed/field_validation_v001` : images,
labels YOLO de tous les ballons visibles, manifeste avec rôle et boîte active,
rapport de complétude et `dataset.yaml` avec `train: null`. Les images exclues ne
sont pas exportées comme négatives. Aucun export de référence n'est créé avant
la revue complète.

## Protection contre la contamination

`configs/reserved_sessions.json` réserve le groupe
`personal_field_session_20261014`, les chemins sources et leurs SHA-256. Le
constructeur de dataset et le lanceur HF vérifient l'absence de ce groupe dans
`train`, y compris au moment de soumettre un entraînement déjà préparé.
L'ajout d'autres clips de cette rencontre exige le même `match_id`.

La configuration d'annotation est `configs/field_validation_20261021.json`.
Le champ historique `role: test_primary` est utilisé uniquement par le transport
d'annotation existant ; le projet de revue et l'export indiquent bien `val` et
`development_validation`. Aucun entraînement n'est lancé par ce workflow.

## Comparaison après validation humaine

Le plan `configs/field_evaluation_20261021.json` fixe les trois checkpoints à
comparer : Nano 960 (Total), Nano 1280 et Small 960. Le premier comparatif utilise
un seuil commun de 0,25 et un appariement unique à IoU ≥ 0,50, séparant :

- la détection de tous les ballons visibles : TP/FP/FN, précision et rappel,
  avec résultats selon la taille du ballon en pixels dans la source ;
- le choix du ballon actif : cible choisie correcte, ballon secondaire choisi,
  fausse cible et absence de cible lorsque l'actif est visible.

Les mesures sur l'actif doivent se limiter aux références avec
`active_identity_reviewed: true`. Une validation des boîtes par Submit ne suffit
pas à revendiquer une validation de l'identité active.

Une boîte sur un ballon de réserve est une bonne détection mais peut être un
mauvais choix pour la caméra. Ne pas mélanger ces deux verdicts. Comparer les
modèles sur les mêmes images, sans régler séparément leurs seuils sur ce lot.
Conserver l'ancien benchmark séparément. Les incertitudes doivent tenir compte
de la corrélation temporelle ; avec une seule rencontre, aucune conclusion sur
la variabilité entre stades n'est possible. Les latences et consommations des
trois modèles doivent être mesurées sur l'iPhone séparément.
