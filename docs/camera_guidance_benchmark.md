# Camera-guidance benchmark on SoccerTrack v2

A closed-loop benchmark of how well a virtual camera, driven by a policy that only sees detected players, keeps the action in frame. It uses three official SoccerTrack v2 MOT clips (240 s each, wide panorama) and was run on 12-13 November 2026. The code is in this repository (`src/box2boxcam/benchmark/`, `src/box2boxcam/camera_sim/`, `scripts/run_camera_benchmark.py`); the dataset itself is not redistributed.

## What it measures

- Data: 3 clips (117092, 117093, 132831) with per-frame boxes of the 22 players, events and ball trajectory (ball used only at event times). Pitch-to-panorama calibration fitted on 65 official points per match (median error 3-8 px).
- 16 segments (577 s) split before any result: 7 tuning, 5 test, 4 final. The test set was looked at twice and the final set once; the final set is consumed.
- A virtual 16:9 camera view moves over the panorama following the policy under test (real closed loop, not a cached replay: a cache gave camera paths 0.18 view-widths apart versus 0.07 between two true runs).
- Metrics: players inside the frame, share of time with fewer than 3 players in frame, distance between frame centre and action centre (densest group of players, smoothed over 1 s) in view-widths, lag at fast transitions and direction reversals per minute.

## Get the data

SoccerTrack v2 (Scott et al., [arXiv:2508.01802](https://arxiv.org/abs/2508.01802), code [AtomScott/SoccerTrack-v2](https://github.com/AtomScott/SoccerTrack-v2), data **CC BY 4.0**) is a gated dataset on Hugging Face: sign in, open the dataset page linked from the GitHub repository, accept the access conditions, and create a read token. Please cite the paper when publishing results. Nothing from the dataset is copied into this repository.

Only three matches (117092, 117093, 132831, release v1.2) are needed. Download them into one folder with the layout below (replace `<repo-id>` by the id shown on the dataset page):

```sh
hf auth login
hf download <repo-id> --repo-type dataset --local-dir soccertrack-v2 \
  --include "mot/117092.txt" "mot/117093.txt" "mot/132831.txt" \
  --include "mot/clips/117092.mp4" "mot/clips/117093.mp4" "mot/clips/132831.mp4" \
  --include "bas/117092/*" "bas/117093/*" "bas/132831/*" \
  --include "ball/*_1st_ball.npz" \
  --include "raw/117092/*_keypoints.json" "raw/117093/*_keypoints.json" "raw/132831/*_keypoints.json"
```

Expected layout (`--soccertrack-root soccertrack-v2`):

| File | Used for |
|---|---|
| `mot/<match>.txt` | per-frame boxes of the players (MOT format, 25 fps, 240 s) |
| `mot/clips/<match>.mp4` | panorama video, first 240 s of the first half (override with `--video-template`) |
| `bas/<match>/<match>_12_class_events.json` | event times |
| `ball/<match>_1st_ball.npz` | ball anchor at event frames (never a per-frame ball truth) |
| `raw/<match>/<match>_keypoints.json` | 65 pitch keypoints for the pitch-to-panorama calibration |

Models (not included, paths are arguments): a COCO person detector (`--people-weights`, e.g. an Ultralytics `yolo26s.pt`), a 17-keypoint pose model (`--pose-weights`, e.g. `yolo26s-pose.pt`) and the ball detector (`--ball-weights`, class 0 = ball). Ultralytics, torch, OpenCV, SciPy and `ffmpeg` are required: `python -m pip install -e '.[benchmark]'`.

## Reproduce

```sh
# one 30 s sanity segment, two variants (about 2-3 minutes on an Apple-silicon laptop)
python scripts/run_camera_benchmark.py --soccertrack-root soccertrack-v2 \
  --set REF --variants current guard --out runs/camera/ref.json \
  --people-weights yolo26s.pt --pose-weights yolo26s-pose.pt --ball-weights ball.pt

# the tuning split with the three headline policies (7 segments)
python scripts/run_camera_benchmark.py --soccertrack-root soccertrack-v2 \
  --set TUNE --variants current guard centroid_motion --out runs/camera/tune.json \
  --people-weights yolo26s.pt --pose-weights yolo26s-pose.pt --ball-weights ball.pt

python scripts/run_camera_benchmark.py --list-variants       # every named variant
python -m box2boxcam.benchmark.calib soccertrack-v2 117093    # calibration leave-one-out error
```

The first variant is the baseline of the paired differences in the output JSON. `--set` is `TUNE`, `TEST`, `FINAL`, `REF` or `ALL`; `--segment-id 117092_a` restricts a split; `--video-template "/videos/{match}.mp4"` points to other video files; `--audit` runs the body-orientation audit (below) instead of scoring. The split is `configs/camera_benchmark_segments.json`, fixed before any result. Runs are sequential and need one process; a 36 s segment takes roughly 1-2 minutes per variant on an M-series laptop. GPU inference is not bit-for-bit deterministic across machines.

Variants (`box2boxcam.benchmark.variants`): `current` (orientation rule), `guard` (+ group guard), `centroid` / `centroid_motion` (centre of the visible players, without body orientation), `cmv5_dz10` (centre + 0.5 s lead + dead zone 0.1), the reference bounds `static` (fixed camera) and `oracle_ac` (camera on the true action centre), and the exploratory ones used during tuning.

A local check on the 30 s `REF` segment (117093, 0-30 s, `current` and `guard`) gave metrics identical to the ones stored when the benchmark was run (same detections, same camera paths): the code move changed no behaviour.

## Metric definitions

Decisions are taken at 10 Hz; the camera crop is the centre plus or minus half of a 16:9 view (a fixed x2 zoom of 0.45 of the panorama width). The first 3 s of each segment are excluded (all variants start at the action centre).

- **Players in frame** (`n_in`): ground-truth players on the pitch whose feet are inside the crop, averaged over decisions. Players are the 22 on the pitch (referee included), feet = bottom-centre of the box, "on pitch" = foot inside the calibrated pitch polygon grown by 1.5 m.
- **Time < 3 players** (`lt3`): share of decisions with fewer than 3 players in the crop.
- **Distance** (`ac_dx`): horizontal distance between the crop centre and the action centre, in view widths. **Action centre**: Gaussian density (sigma 8 m) of the on-pitch players, evaluated at each player; the centre is the mean position of the players within 15 m of the densest one, smoothed over 1 s. It is the densest group of players, not the ball.
- **Lag**: at fast transitions (the action centre moves at least 0.6 view widths within 3 s), the time between the action centre and the crop centre each covering half of the move (8 s if never reached).
- **Reversals**: x-direction reversals of the crop per minute, ignoring pan speeds below 0.05 view widths per second.
- Also reported: share of event times with the action centre / ball anchor inside the crop (`ev_ac_in`, `ev_in`), mean pitch error at events (`ev_m`), mean pan speed (`speed`), share of decisions with a ball measurement (`ball_used`).
- Aggregation: mean over segments with a 95 % percentile bootstrap over segments (2,000 resamples, seed 0); paired differences against the first variant use the same segments.

## Headline numbers (tuning / test / final)

| Policy | Players in frame | Time < 3 players | Distance | Lag vs current (s) | Reversals |
|---|---|---|---|---|---|
| Current rule | 14.5 / 13.2 / 11.8 | 0.6 / 4.5 / 2.2 % | 0.253 / 0.312 / 0.334 | reference | 15.0 / 15.0 / 16.5 per min |
| + group guard | 15.4 / 13.8 / 12.0 | 0.2 / 1.4 / 1.7 % | 0.209 / 0.274 / 0.347 | -0.16 / -0.07 / +0.08 | +12 / +7 / +4 % |
| Centre of players | 15.8 / 13.8 / 13.7 | 0.4 / 1.4 / 1.4 % | 0.202 / 0.349 / 0.203 | +0.58 / +3.3 / +0.52 | +34 / -2 / -3 % |
| Centre + 0.5 s lead + dead zone 0.1 | 15.6 / 14.2 / 13.3 | 0.6 / 1.4 / 1.5 % | 0.210 / 0.276 / 0.234 | -0.06 / +1.9 / +0.59 | -1 / -38 / -6 % |

Values are tuning / test / final. Lag and reversals are relative to the current rule. The test set was looked at twice, the final set once.

Reference points on the final set: a fixed camera scores 0.83 (29 % of the time under 3 players); a camera placed on the true action centre scores 0.056.

## Group guard: result and caveat

The group guard keeps the camera with the densest group of players instead of parking on a single one. Pooled over the 16 segments the distance improves by 0.028 [95 % CI -0.055, -0.001] and the share of time under 3 players falls from 2.2 % to 1.0 %. It removes the "camera parked on one player" episodes (15 % to 0 % on one 38 s segment). Constants (see `box2boxcam.camera_sim.planner`): 4 s window; enter below 0.40 x peak, leave at 0.60 x peak; minimum peak 5 players; return rate 0.30 view widths/s, at most 6 s. **Caveat:** on the final set alone there is no gain (+0.013 [-0.03, +0.08] in distance), so the gain is modest and not confirmed on held-out segments.

## Body orientation is a weak group-direction signal

Among 4,463 players moving at 2.5 m/s or more:

- 42 % get a single heading estimate from body pose: median error 12 degrees, 78 % within 45 degrees, but 15 % point backwards. The torso axis itself is accurate (11 degrees).
- The shared direction the camera actually follows has the right sign along x only 60 % of the time (median error 41 degrees). Correcting forward/backward with displacement made framing worse.
- A follow-up replaced body orientation by tracked player velocity. The fresh signal is better (right sign 93 %, 6 degrees) but fires three times as often and describes a running sub-group, so the camera overshoots ahead of the dense group. On the tuning set no variant beat the group guard (distance +0.07 to +0.23 worse), so none was taken to the test set.



## Off-pitch spare-ball filter (no calibration)

Spare balls outside the pitch are a source of false positives. Rejecting a candidate that is static and outside the convex hull of detected players' feet (expanded by one median player height, at least 3 players in view) rejected 31 % of outside detections in a realistic camera view (68 % with the full panorama) with 0 of 116 in-play detections lost on the test look. It is a mitigation, not a replacement for a pitch polygon.

## Caveats

- **The FINAL split is consumed**: it was looked at once, at the end, for the published numbers. Re-running it is a reproduction, not a held-out test; any new policy needs fresh segments (and ideally other matches).
- **Only 21 fast transitions** in total: lag numbers are noisy and their confidence intervals cross zero.
- **The "action centre" is the densest group of players**, not the ball (SoccerTrack v2 has no per-frame ball truth; the ball track is reconstructed between annotated events). A camera that follows the ball elsewhere is scored as being away from the action.
- The final set is four segments of the same three matches as tuning and test; the gain of the group guard is modest and not confirmed on it.
- Detections run on the camera's own crop (closed loop). A cached full-panorama replay was tried first and rejected: it gave camera paths 0.18 view-widths apart versus 0.07 between two true runs.
- `--audit` replays the orientation audit with closed-loop rows; the 4,463-player figure above came from an earlier replay of cached full-panorama detections, so counts differ slightly.
- Not included: the video rendering tools, the full-panorama detection cache and the later velocity-heading experiments (negative result, described above).
