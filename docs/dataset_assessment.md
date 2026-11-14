# Initial dataset assessment

Assessment of the public football-ball datasets considered for box2boxcam (state as of 15 September 2026), the cleaning applied to Infactory, and the resulting `v001` and `v002` working datasets. Infactory is the best starting point after cleaning; none of the sources matches the real target (a fixed iPhone on an amateur touchline).

## Short conclusion

The best available starting point is **Infactory**, after cleaning. It has consecutive sequences, many true negatives and enough annotated balls for a first baseline. **SoccerEye** can complement training with much smaller balls, but its published splits are unreliable and its data license is not stated.

No dataset represents the target. They allow us to learn and build the pipeline, but final validation must use our own videos.

Because the project is currently non-commercial, research-only and non-commercial sources are usable. This does not remove attribution duties, redistribution bans, or the uncertainty of a missing license.

## Decision per source

| Source | Immediate usefulness | Current decision | Main reason |
|---|---|---|---|
| Infactory Ball Tracking | High | Use after cleaning | 7,152 temporal frames, but 75 invalid annotations and a TV domain |
| SoccerEye | Medium | Keep locally, rebuild splits | Very small balls, but only 427 boxes and leakage between splits |
| Roboflow Football Ball v4 | Medium | Wait for authenticated access | 4,948 tiles under CC BY 4.0, from 1,237 parent images |
| SoccerNet Tracking | High for tracking | Later, after NDA | Very rich temporally, heavy and research-only |
| SoccerNet v3 H250 | Potentially high | Blocked by license | Interesting long-distance subset, license not displayed |
| Dryad Sport Ball 2026 | Uncertain | Blocked | Small annotation package, external images/videos not retrieved |
| box2boxcam captures | Essential | To be created later | Only source that matches the target product |

License terms and source links are centralized in `configs/data_sources.toml`.

## 1. Infactory Ball Tracking

### What is actually present

[The Hugging Face card](https://huggingface.co/datasets/infactory-ai/ball-tracking) announces 1,450 frames from 10 clips. The downloaded revision `75854a5837dd3302cbb1843355ecc1a6f124f818` actually contains:

- 7,152 data images;
- 54 complete UUID sequences, numbered without gaps;
- sequences of 15 to 588 frames, median 82;
- 4,232 images with at least one valid box;
- 2,845 images with no annotated ball;
- 75 images with a zero-area annotation;
- 4,298 valid boxes in total.

`metadata.csv` still describes only the old 1,450-image subset, so it must not be used as a catalog of the current download.

### Ball size

At native resolution the ball box has a median of about `18.0 x 16.2 px`. After proportional resize and letterbox to `640 x 640`, it is about `13.8 x 12.5 px`. The hardest tenth drops to about `7.5 x 6.5 px` at 640: difficult but workable for a first experiment. The two resolutions present are `1280 x 720` and `720 x 608`.

### Quality problems

- A 50-frame sequence marked `visible` is a TV title card with `0 0 0 0` boxes.
- Three other sequences contain 25 zero-area annotations, i.e. 75 invalid images across four sequences.
- 63 images have several boxes although a normal frame should contain one ball. The four-box example contains at least three visibly wrong annotations, so all 63 need human review.
- Images come from professional broadcasts: varied angles, close-ups, slow motion, interlacing and watermarks. This is not the amateur-camera domain we target.

### Chosen treatment

`data/raw/infactory_ball_tracking` is never modified. The working version `data/processed/v001`:

1. excludes the 75 invalid annotations;
2. excludes the 63 multi-box frames pending correction;
3. splits train/validation/test by full UUID;
4. reserves whole sequences for test;
5. keeps neighboring frames of a sequence together;
6. keeps the 138 exclusions in `review_queue.csv`.

This syntactic cleanup was not enough. A stratified visual QA found single boxes on objects that are not the ball, and a negative frame where the goalkeeper visibly holds the ball.

## 2. SoccerEye

### Measured content

- 1,128 images and 28,755 boxes;
- class `0` ball: 427 boxes in 414 images;
- class `1` person: 28,328 boxes across all 1,128 images;
- 714 images contain no ball and become negatives for a ball-only detector;
- 911 images at `640 x 640`, 217 at `1920 x 1080`.

The median ball is `6 x 9 px` natively, dropping to about `4 x 6.5 px` after letterbox to 640. The dataset is therefore useful for the genuinely hard tiny-ball case.

### Leakage between published splits

File names encode a Roboflow parent identifier. The audit finds:

- only 560 parent names for 1,128 files;
- 128 parents present once, 296 twice and 136 three times;
- 53 parent names present in several splits;
- 15 presumed source prefixes, all present in at least two splits;
- no strictly identical file (SHA-256) across splits.

The copies are not byte-identical, but the names show that variants of the same parent image were distributed between training and evaluation. Splits must be rebuilt before trusting any metric.

The MIT license of the [SoccerEye repository](https://github.com/WWandP/SoccerEye) covers the code. No explicit license was found for the dataset linked via Google Drive. It can be studied locally, but must not be republished on Hugging Face until this is clarified.

## 3. Sources not downloaded

### Roboflow Football Ball Detection v4

[Version 4 on Roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg/dataset/4) lists 4,948 generated images: 3,956 train, 492 validation and 500 test. Preprocessing cuts each of the 1,237 parent images into four tiles, then stretches them to `640 x 640`. Automated download requires Roboflow authentication.

If added, splitting must happen at parent-image level **before** tiling, otherwise zones or variants of the same image could contaminate evaluation.

### SoccerNet Tracking

[SoccerNet Tracking](https://www.soccer-net.org/data) is especially interesting for learning temporal tracking. The benchmark announces 201 sequences, 225,375 frames and 215,156 ball boxes. Videos are professional, `1920 x 1080`, and available after signing a research NDA. It is not downloaded at this stage: it is heavier and less direct than Infactory for a first baseline.

### Dryad and SoccerNet v3 H250

The [Dryad football package](https://datadryad.org/dataset/doi:10.5061/dryad.3bk3j9m13) is only about 3.27 MB and seems to contain annotations, with source videos hosted separately. Automatic download is currently refused.

The [SoccerNet v3 H250 subset](https://zenodo.org/records/7808511) contains 19,786 long-distance images with ball/person boxes. It is promising, but the Zenodo record shows no explicit license and derives from SoccerNet, so it stays blocked until clarified.

### Video behind the 1,120 YouTube frames

`data/external/videos/inference_input_yt.mp4`, the source of the 1,120 frames (dataset id `luna_v001` in configs), was downloaded from <https://www.youtube.com/watch?v=vL0r9ek1nqQ>. This origin is stated by the project owner and has not been re-verified. Redistribution rights are not cleared: annotations and the URL may be published, frames and the video may not. Recorded in `configs/data_sources.toml` and `configs/video_download_sources.json`.

## Scientific rules for box2boxcam

1. **A source group never crosses splits.** For video, the group is the video or match, never the frame.
2. **The final test stays human and clean.** No teacher pseudo-label is accepted as test ground truth.
3. **Teacher and student share train, not test.** The large model can propose training labels; a human decides ambiguous cases.
4. **Small balls are measured separately.** Results are broken down by size after resize (`< 4 px`, `4-8 px`, `8-16 px`, `> 16 px`).
5. **Negatives are kept.** A camera must stay stable when the ball is hidden or out of frame.
6. **The target decides.** A small collection of matches filmed at the same place as a future gimbal will be the real feasibility test.

## Working dataset `v001`

Notebook `02_dataset_preparation.ipynb` built a first conservative version limited to Infactory:

- 7,014 images kept: 4,169 positive and 2,845 negative;
- 138 images excluded and kept in a review queue;
- 38 sequences and 4,911 images in train;
- 8 sequences and 1,071 images in validation;
- 8 sequences and 1,032 images in test;
- no sequence shared between two splits;
- no invalid box or multi-box image in the prepared version;
- 7,014 unique SHA-256 hashes, so no exact duplicate across splits.

### Visual QA result

The notebook shows 20 deterministic examples per split: four balls for each of the `tiny`, `small`, `medium`, `large` categories, then four negatives. Across these 60 images, five certain errors were logged: four false positives and one false negative.

Visual validation of `v001` therefore **failed**, even though mechanical validation passed. The notebook produces a priority queue of 700 images:

- 508 in train;
- 48 in validation;
- 144 in test;
- 589 large boxes to check;
- 86 isolated negatives between two positive frames;
- 29 abrupt jumps in position or area, with some overlap between rules;
- the five errors already confirmed visually.

These signals are not automatic decisions: a large box may be a genuine close-up and a jump may be a camera cut. They only order the human review (`data/processed/v001/qa_review_queue.csv`).

Before the nano baseline, at least the 192 validation and test candidates must be cleaned and `v002` generated without modifying `raw` or `v001`. SoccerEye will come in a separate experiment, to measure its contribution on very small balls without masking Infactory's defects.

For the actual review, `03_dataset_review.ipynb` merges these 700 candidates with the 138 already-excluded annotations. The full human queue is therefore 838 images: 600 train, 57 validation and 181 test. The earlier counts of 48 validation and 144 test only cover annotations that had been accepted in `v001`; the 37 validation/test exclusions are added in the correction interface.

### Extension to an exhaustive review

The 838-candidate queue was a selection, not an exhaustive check. After errors found during review, `03_dataset_review.ipynb` now shows all 7,152 Infactory source images, including negatives with no automatic signal and the initial exclusions. Earlier decisions are kept. The full queue holds 1,069 test, 1,080 validation and 5,003 train images, ordered by split, sequence and frame. Signals explain risks rather than restrict coverage. SoccerEye remains separate.

## `v002` built (16 September 2026)

Rule: no decision means keeping the original label. Several true balls are allowed. Explicit corrections replace the label; individual and grouped exclusions remove the image.

| Split | Images | Positive | Negative | Boxes | Sequences |
|---|---:|---:|---:|---:|---:|
| Train | 3,048 | 2,113 | 935 | 2,114 | 30 |
| Validation | 670 | 494 | 176 | 494 | 7 |
| Test | 831 | 639 | 192 | 651 | 8 |
| Total | 4,549 | 3,246 | 1,303 | 3,259 | 45 |

Actions: 2,913 labels kept by default, 1,298 explicit confirmations, 246 corrections, 92 switches to negative, 2,603 exclusions. Thirteen multi-box images are kept (one in train and twelve in test). The structural audit is valid; no sequence or exact duplicate crosses splits. Proportions shifted during triage because sequences were not reassigned.

This version was triaged by hand with keep-by-default; it is not fully re-annotated ground truth, so errors may remain in the original labels, including in test. Raw sources and `v001` remain available. The next milestone is the first reference training.
