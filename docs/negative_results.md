# What did not work

Experiments that did not improve the ball detector or the camera policy. All numbers are on the reserved match (396 frames, 219 balls of which 149 are small, IoU >= 0.5, 960 px input).

- **WASB published weights** (soccer checkpoint, run on the 396 reserved frames in two input variants, scored by centre-in-box with 4 px tolerance). Variant A finds 27 of 219 balls at its loosest threshold (0.10) with 2,287 false positives; variant B finds 87 with about 26,000 false positives, versus 101 true / 102 false positives for v006 at 0.25. Small balls: 5 and 45 of 149, versus 58 for v006. Used as a second detector it adds almost nothing at a usable false-positive level.
- **3-frame stacking (9-channel input).** A second attempt, with a matched 3-channel control, did not beat the control beyond noise: the sign of the difference flipped with the control checkpoint (+9 versus -7 to -9 true positives), neither beat v006 at 25 or 60 false positives, and an epoch cost about 3x the GPU time.
- **Gated low-threshold re-detection and frame-difference re-scoring.** Replayed offline: +1 to +6 true positives per recording with confidence intervals including zero net gain. About 80 % of misses produce no candidate at all, so no re-scoring can recover them. Side finding: raising the camera policy's candidate floor from 0.10 to 0.25 cut chosen-ball false positives by roughly 60 % in the replay (to be confirmed on device).
- **Copy-paste of small balls.** Pasting small balls into training images gave no gain (94/103 TP/FP at 0.25 versus 100/103 for v006).
- **A single-seed "gain" that did not replicate.** A mosaic + reduced-scale recipe (model B) looked like a precision gain (-39 false positives at 0.25 versus its control). Re-running the same recipe with another seed (model D) gave +59 false positives at the same threshold, and at equal false positives the difference to v006 stayed inside the noise. The earlier gain was mostly seed and calibration luck, so v006 stays the reference. Longer training and a more aggressive scale range were not better either.
- **Body orientation for group direction.** See the [camera-guidance benchmark](camera_guidance_benchmark.md).

Recall on small, far balls is still below 50 %: v006 finds 57 of the 149 small balls at a 0.25 threshold.
