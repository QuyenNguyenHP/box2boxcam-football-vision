# Camera-guidance benchmark on SoccerTrack v2

A closed-loop benchmark of how well a virtual camera, driven by a policy that only sees detected players, keeps the action in frame. It uses three official SoccerTrack v2 MOT clips (240 s each, wide panorama) and was run on 12-13 November 2026. The benchmark scripts live in an internal working folder and are not part of this repository; this page reports the protocol and the numbers.

## What it measures

- Data: 3 clips (117092, 117093, 132831) with per-frame boxes of the 22 players, events and ball trajectory (ball used only at event times). Pitch-to-panorama calibration fitted on 65 official points per match (median error 3-8 px).
- 16 segments (577 s) split before any result: 7 tuning, 5 test, 4 final. The test set was looked at twice and the final set once; the final set is consumed.
- A virtual 16:9 camera view moves over the panorama following the policy under test (real closed loop, not a cached replay: a cache gave camera paths 0.18 view-widths apart versus 0.07 between two true runs).
- Metrics: players inside the frame, share of time with fewer than 3 players in frame, distance between frame centre and action centre (densest group of players, smoothed over 1 s) in view-widths, lag at fast transitions and direction reversals per minute.

## Headline numbers (tuning / test / final)

| Policy | Players in frame | Time < 3 players | Distance |
|---|---|---|---|
| Current rule | 14.5 / 13.2 / 11.8 | 0.6 / 4.5 / 2.2 % | 0.253 / 0.312 / 0.334 |
| + group guard | 15.4 / 13.8 / 12.0 | 0.2 / 1.4 / 1.7 % | 0.209 / 0.274 / 0.347 |
| Centre of players | 15.8 / 13.8 / 13.7 | 0.4 / 1.4 / 1.4 % | 0.202 / 0.349 / 0.203 |

Reference points on the final set: a fixed camera scores 0.83 (29 % of the time under 3 players); a camera placed on the true action centre scores 0.056.

## Group guard: result and caveat

The group guard keeps the camera with the densest group of players instead of parking on a single one. Pooled over the 16 segments the distance improves by 0.028 [95 % CI -0.055, -0.001] and the share of time under 3 players falls from 2.2 % to 1.0 %. It removes the "camera parked on one player" episodes (15 % to 0 % on one 38 s segment). **Caveat:** on the final set alone there is no gain (+0.013 [-0.03, +0.08] in distance), so the gain is modest and not confirmed on held-out segments.

## Body orientation is a weak group-direction signal

Among 4,463 players moving at 2.5 m/s or more:

- 42 % get a single heading estimate from body pose: median error 12 degrees, 78 % within 45 degrees, but 15 % point backwards. The torso axis itself is accurate (11 degrees).
- The shared direction the camera actually follows has the right sign along x only 60 % of the time (median error 41 degrees). Correcting forward/backward with displacement made framing worse.
- A follow-up replaced body orientation by tracked player velocity. The fresh signal is better (right sign 93 %, 6 degrees) but fires three times as often and describes a running sub-group, so the camera overshoots ahead of the dense group. On the tuning set no variant beat the group guard (distance +0.07 to +0.23 worse), so none was taken to the test set.

Uncertainties: only 21 fast transitions (noisy lag), the "action centre" is the densest player group rather than the ball, and the final set is four segments of the same matches.

## Off-pitch spare-ball filter (no calibration)

Spare balls outside the pitch are a source of false positives. Rejecting a candidate that is static and outside the convex hull of detected players' feet (expanded by one median player height, at least 3 players in view) rejected 31 % of outside detections in a realistic camera view (68 % with the full panorama) with 0 of 116 in-play detections lost on the test look. It is a mitigation, not a replacement for a pitch polygon.
