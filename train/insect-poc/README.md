# Insect POC

This experiment trains a collection of small RBF-SVM specialists and exports
portable numeric artifacts for low-overhead CPU inference. Specialists share an
ensemble contract, not a training lifecycle: replacing `motion` never retrains
or overwrites `presence`, `color`, or `gesture`.

`InsectEnsemble.compose_observation()` converts accepted specialist results into
bounded VLM-like text such as `Person detected, moving left, wearing red.`. It
never invents a property rejected as `unknown`.

## Quick experiment

```bash
./train/insect-poc.sh all
./train/insect-poc.sh list
```

## Genuine recurrent-core experiment

The recurrent test consumes at most 15 minutes of grouped temporal data, trains
a fixed 256-unit core, and automatically trains a matched zero-recurrence
ablation. It issues `PASS` only when recurrent held-out macro-F1 improves by more
than 0.5 points without increasing temporal flicker.

First validate the machinery with deliberately noisy temporal contract data:

```bash
./train/insect-poc.sh prepare-core-demo
BACKEND=mlx ./train/insect-poc.sh train-core
cat train/insect-poc/runs/shaped-core/report.json
```

For a genuine camera result, replace `data/core_sequences.npz` with:

- `X`: float32 `[frames, features]`, ordered in time;
- `y`: int32 state (`0=absent, 1=present_still, 2=present_left, 3=present_right`);
- `recording_id`: int32 recording/clip ID; splits and state resets use it;
- `timestamp_s`: float32 time within the recording;
- `fps`: scalar integer.

Frames from one recording never cross train/validation/test boundaries. The
trainer robustly normalizes from training frames only, constrains recurrent
spectral radius to 0.85, uses leak state, class-weighted loss, gradient clipping,
early stopping, explicitly detects single-class collapse, and saves
recurrent/no-recurrence artifacts separately. Demo data includes transient
dropouts, illumination events, camera shake, and adversarial single-frame
specialist errors. It remains a temporal contract test rather than camera proof.

### Genuine IPN Hand evidence

The bounded fetch downloads official annotations and one of five ~1GB MP4
shards—not the full 4.6GB corpus. After extraction, the archive is removed to
respect constrained development disks.

```bash
./train/insect-poc.sh fetch-ipn
./train/insect-poc.sh prepare-ipn-core
BACKEND=mlx ./train/insect-poc.sh train-core
```

`prepare-ipn-core` samples at 15 FPS, stops at 15 minutes, keeps recording IDs,
and derives 12 cheap contour/motion/illumination features. Ground truth comes
from IPN's independent frame annotations: `D0X=no_gesture`, `G05=throw_left`,
`G06=throw_right`, and other hand activity=`present_other`. A provenance
manifest is written beside the NPZ. IPN Hand data and annotations are CC BY 4.0.

HaGRID and `Vincent-luo/hagrid-mediapipe-hands` are static-image sources, so they
can strengthen gesture specialists but cannot prove recurrent improvement.
The latter is 112GB and is intentionally not materialized by this bounded
workflow. NVIDIA Dynamic Hand Gesture should be added as a second held-out video
domain after the IPN vertical slice; mixing it into the first train/test split
would make it harder to identify basic pipeline failures.

### MediaPipe trajectory direction specialist

```bash
./train/insect-poc.sh prepare-ipn-direction
BACKEND=mlx ./train/insect-poc.sh train motion
./train/insect-poc.sh assemble
./train/insect-poc.sh bench motion
```

This extracts wrist and index-tip displacement over 1-, 5-, and 15-frame
horizons, normalized by palm scale. Labels come independently from
IPN annotations: `G03=up`, `G04=down`, `G05=left`, `G06=right`; all remaining
activity is `still`. Recording IDs are retained for group-held-out validation.

`BACKEND=auto` tries MLX/Metal first, RAPIDS cuML/CUDA second, and scikit-learn
last. On an Apple-silicon trainer, require—not merely prefer—the M4 GPU with:

```bash
BACKEND=mlx ./train/insect-poc.sh train all
```

Retrain and atomically replace only one specialist, then refresh the manifest:

```bash
BACKEND=mlx ./train/insect-poc.sh train motion
./train/insect-poc.sh assemble
./train/insect-poc.sh bench motion
```

The previous specialist is retained as `runs/specialists/<name>.previous`.

MLX does not provide a packaged SVC solver. The MLX backend constructs the RBF
kernel matrix and optimizes a class-balanced squared-hinge kernel objective on
Metal, using a bounded candidate-support set. Small coefficients are pruned and
the result is exported to the same support-vector decision function as the
cuML/libsvm backends. `max_support_vectors`, `mlx_epochs`, and
`mlx_learning_rate` are independently configurable for each specialist.

## Commands

- `list`: show configured specialists and training state.
- `prepare-demo [names...]`: generate deterministic contract/smoke data.
- `train [names...]`: train only the named specialists (`all` is the default).
- `assemble`: validate existing specialists and write `runs/ensemble.json`.
- `bench [names...]`: benchmark portable CPU inference and record model size.
- `all [names...]`: prepare demo data, train, assemble, and benchmark.

Useful environment variables are `BACKEND`, `PYTHON`, `CONFIG`, `DATA_DIR`,
`RUNS_DIR`, `SAMPLES`, `ITERATIONS`, and `SEED`.

## Real feature data contract

Replace `data/<specialist>.npz` with an extractor's output:

- `X`: `float32 [samples, feature_count]`
- `y`: string class labels matching `config.json`
- `groups`: optional recording/subject IDs used for leakage-resistant validation

The `feature_extractor` identifier and `feature_count` in `config.json` are the
compatibility boundary. Change either for one specialist, regenerate only that
NPZ, and retrain only that specialist.

The included demo generator proves orchestration and portability; its accuracy
is not evidence of camera performance. Real energy comparisons must replay the
same frames and outputs for both systems. The benchmark records real RAPL joules
when Linux exposes them; otherwise it reports latency, throughput, CPU time, and
artifact bytes without inventing an energy estimate. Use an inline USB power
meter for defensible Raspberry Pi joules/inference.
