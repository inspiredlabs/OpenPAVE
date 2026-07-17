# Pixel Sensor Stage-1 Spike

This is the bounded Rev-2 experiment from
[`docs/pixel-sensor-module-spec.md`](../../docs/pixel-sensor-module-spec.md):
the existing v3 trunk proposes 21 two-dimensional points and a 35,175-parameter
patch model either corrects or rejects each proposal. It has no MediaPipe
runtime dependency and makes no metric-3D claim.

## Contracts implemented

- `projector.py`: two-dimensional Umeyama rotation/scale/translation solve,
  inverse transform, degeneracy rejection, and sub-pixel round-trip tests.
- `train.py`: per-joint visibility handling, trunk-distribution positives,
  adjacent-joint/background negatives, source-held validation, frozen-threshold
  YOLO26 referee, MPS training, and ONNX export.
- `runtime.py`: CPU ONNX patch sensing, confidence collapse, recurrent passes,
  and a coordinate adapter whose provenance explicitly says
  `xy_source=pixel_patch`, `z_source=prior`, `metric_3d=false`.
- `palm_decoder.py`: exact 2,016-anchor generation, 18-value box/seven-point
  decoding, sigmoid filtering, weighted IoU suppression, letterbox removal,
  and rotation-aligned hand ROI construction.
- `benchmark.py`: actual single-frame CPU timings; it does not report the
  highly amortised training evaluator as runtime latency.

Run:

```bash
./train/pixel-sensor.sh test
./train/pixel-sensor.sh palm-inspect
./train/pixel-sensor.sh palm-smoke
./train/pixel-sensor.sh train
./train/pixel-sensor.sh benchmark
```

Artifacts are written to `train/runs/pixel_sensor/`.

## Frozen first result

The confidence threshold was selected using only the exploration holdout, then
YOLO26 was opened once as the foreign referee.

| split | trunk mean | refined mean | relative change | trunk/refined p95 |
|---|---:|---:|---:|---:|
| exploration holdout | 78.61 px | 73.25 px | 6.8% better | 174.39 / 176.17 px |
| YOLO26 referee | 68.11 px | 64.26 px | 5.7% better | 143.31 / 149.60 px |

Pixels are reported at 384-equivalent resolution. One recurrent pass is the
frozen result. Two and three passes improved the familiar holdout but reduced
foreign mean performance and enlarged its p95 tail, so recurrence is not
enabled by default.

Verdict: **the narrow thesis passes**—local patches contain corrective signal
that beats the trunk's own mean error on both splits. **Runtime promotion is
blocked**—absolute error is far above the specification's 5 px target and the
p95 tail regresses. This artifact must not appear in the GUI controller list.

Actual single-frame ONNX Runtime CPU measurements on the M4 are:

| component | median | p95 |
|---|---:|---:|
| v3 trunk | 3.04 ms | 5.92 ms |
| one 21-patch pass | 0.89 ms | 2.22 ms |
| combined | 4.07 ms | 8.17 ms |

These measurements exclude the ROI detector, Monty matching, GUI and camera
capture. The patch network meets its local cost objective reasonably well,
but the trunk-initialised stack does not meet the 2 ms end-to-end target.

## BlazePalm cold-start finding

The bundled `hand_detector.tflite` was inspected directly through its TFLite
flatbuffer. Its tensors are:

```text
input:  [1, 192, 192, 3]
output: [1, 2016, 18] regressors
output: [1, 2016, 1]  scores
```

The 18-value regressor is four box values plus seven 2D palm keypoints. Anchor
generation and ROI decoding are now implemented and pinned to the official
MediaPipe graph constants. The remaining runtime seam is raw TFLite execution:
the current macOS environment exposes MediaPipe's complete task runner but no
standalone interpreter API for the two raw tensors. The decoder therefore
accepts those tensors explicitly and is ready for LiteRT/TFLite on the target;
it is not yet claimed as camera-validated on this Mac.

The standalone macOS interpreter lives at
`/opt/anaconda3/envs/openpave-tf`. A new shell exposes
`$OPENPAVE_TF_PYTHON` and the `activate-openpave-tf` alias. `palm-smoke`
executes the bundled detector with that interpreter and sends its real raw
tensors through this decoder.
