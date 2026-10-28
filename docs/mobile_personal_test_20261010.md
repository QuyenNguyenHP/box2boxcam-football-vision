# Nano mobile 960 — test personnel final

## Protocole

Comparaison du précédent nano Luna et du nouveau nano mobile, tous deux à
960 px sur les 292 images revues du clip court du 30 septembre. Les modèles
et le seuil étaient figés avant le test. Aucune modification d'entraînement,
de labels ou de seuil n'a été décidée à partir de ces résultats.

- 264 images positives, 28 négatives, 279 boîtes ; 11 images multiboîtes.
- 88 frames échantillonnées exclues par la revue ne sont pas évaluées.
- Confiance ≥ 0,25. Détection stricte : IoU ≥ 0,50.
- CPU, batch 1 et traitement séquentiel pour limiter la mémoire du Mac.
- **Limite : train long et test court viennent du même match.** Le test mesure
  un autre clip de cette situation, pas la généralisation à un match inconnu.

## Détection de toutes les boîtes

| Mesure | Ancien nano | Nouveau nano |
|---|---:|---:|
| mAP50 | 5,5 % | 65,4 % |
| mAP50–95 | 1,3 % | 33,1 % |
| Précision | 11,0 % | 62,9 % |
| Rappel | 15,8 % | 65,6 % |
| F1 | 12,9 % | 64,2 % |
| TP | 44 | 183 |
| FP | 357 | 108 |
| FN | 235 | 96 |
| FP sur les 28 images négatives | 32 | 15 |

Les mAP intègrent plusieurs seuils ; précision/rappel/F1 et TP/FP/FN utilisent
le point fixe ci-dessus. La matrice de confusion personnalisée utilise ces
mêmes seuils ; sa case de vrais négatifs est vide, car le fond n'est pas un
ensemble dénombrable d'objets négatifs.

## Cadrage : une seule cible

On choisit la boîte la plus confiante, sans regarder l'annotation, puis on
applique le seuil 0,25. Une mauvaise cible sur une frame positive compte comme
une erreur de sélection et un ballon manqué. Une bonne abstention sur une
frame négative compte comme une bonne décision.

| Décision par frame, IoU ≥ 0,20 | Ancien nano | Nouveau nano |
|---|---:|---:|
| Bonne cible sur frame positive | 40 | 182 |
| Mauvaise cible sur frame positive | 150 | 23 |
| Aucune cible sur frame positive | 74 | 59 |
| Fausse alerte sur frame négative | 17 | 10 |
| Bonne abstention sur frame négative | 11 | 18 |
| Bonnes décisions sur 292 frames | 17,5 % | 68,5 % |
| F1 de ciblage | 17,0 % | 76,0 % |

Le résultat est identique à IoU ≥ 0,10 pour ces prédictions. Les prédictions
normalisées de ciblage sont enregistrées dans `predictions.json` ; elles
utilisent une inférence carrée à 960, commune aux deux modèles. La validation
de détection Ultralytics utilise son prétraitement de validation. Ce sont
deux protocoles explicites, pas un simple recalcul de la mAP au seuil 0,20.

Avec plusieurs annotations, **n'importe quel ballon annoté peut correspondre**.
Ce benchmark ne sait pas identifier le ballon actif parmi ceux des autres
équipes. Il ne mesure pas non plus un tracker ni la commande d'un gimbal.

## Vidéo et fichiers

La vidéo complète montre le nouveau nano sur les 1 517 frames du clip, à la
cadence moyenne de la source. Elle comprend aussi des frames non annotées ou
exclues ; aucune métrique supplémentaire n'est déduite de leur présence.

Une seule cible est affichée, avec un contour vert fin et une marge visuelle
de deux pixels. La confiance est au-dessus de l'image, pas sur le ballon.
Pas de tracker, lissage ou correction humaine des prédictions. Le MP4 H.264
conserve la piste AAC standard ; la piste audio spatiale APAC du MOV n'est
pas exportée. Les images du terrain restent à leur résolution originale.

- `models/reports/mobile_personal_test_20261010/report.json` : métriques, hashes
  des poids et du manifeste, décisions et informations vidéo.
- `models/reports/mobile_personal_test_20261010/predictions.json` : les 292
  images de test avec les boîtes des deux modèles.
- `models/reports/mobile_personal_test_20261010/lou_mistrau_nano960_ball.mp4` :
  visualisation du clip complet.
- `notebooks/09_mobile_training.ipynb` : résultats de validation, apprentissage,
  test personnel et vidéo, sans lancer de nouveau job payant.

L'évaluation se reproduit dans un nouveau dossier avec :

```bash
python scripts/evaluate_mobile_personal.py --output models/reports/nouveau_test
```

`--video-only` reprend uniquement la vidéo si les métriques sont déjà figées,
en vérifiant que les poids n'ont pas changé. Il ne relance pas le test.

Le gain est important sur ce clip, mais 59 frames positives restent sans cible,
23 ciblent une mauvaise boîte et 10 négatives déclenchent une alerte. Une vidéo
d'un autre match/date est nécessaire pour mesurer une généralisation plus
indépendante. Aucun nouveau job GPU ou appel Luna payant n'a été lancé pour
cette évaluation.
