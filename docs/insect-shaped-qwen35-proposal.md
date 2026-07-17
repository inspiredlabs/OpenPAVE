# Proposal: Insect-Shaped Qwen3.5-2B Vision-Language System

## Status

Experimental architecture proposal for OpenPAVE. The purpose is to test whether
a fixed, recurrent, specialist-driven visual substrate can preserve the useful
illusion of continuous scene understanding while running a 2B VLM only when the
small system is uncertain or encounters novelty.

This proposal adapts the architectural principle in *Biological Processing
Units: Leveraging an Insect Connectome to Pioneer Biofidelic Neural
Architectures* rather than claiming to reproduce its biological model. The
paper converts the approximately 3,000-neuron, 65,000-edge Drosophila larval
connectome into a fixed recurrent core. It partitions neurons into sensory,
internal, and output pools, leaves recurrent connectivity fixed, and trains
input/output projections around it. The reported results demonstrate parameter
efficiency on MNIST, CIFAR-10, and ChessBench, but do **not** by themselves prove
lower wall-plug energy for live video. That remains an OpenPAVE measurement
question. [Paper](https://arxiv.org/abs/2507.10951),
[readable summary](https://www.emergentmind.com/papers/2507.10951)

Qwen3.5-2B is an Apache-2.0 multimodal model with a vision encoder, a 2B language
model, 24 layers, hidden width 2,048, and a hybrid layout combining Gated
DeltaNet and gated attention. The official model card positions it for
prototyping and task-specific fine-tuning. This proposal keeps that model intact
at first; it changes what reaches it and how often it runs.
[Official model card](https://huggingface.co/Qwen/Qwen3.5-2B)

## Hypothesis

A conventional VLM repeatedly converts nearly identical video frames into
visual tokens and performs autoregressive decoding. Most of those frames contain
no new semantic information. A shaped system can instead:

1. continuously evaluate cheap specialist features;
2. integrate them through a small fixed recurrent state;
3. emit deterministic observations for familiar, high-confidence states;
4. call Qwen3.5-2B only for ambiguity, novelty, or an explicit open-vocabulary
   question; and
5. distil useful Qwen answers back into specialist training data.

The intended win is therefore **VLM duty-cycle reduction**, not a claim that a
small RBF ensemble has become an open-vocabulary VLM.

## Proposed architecture

```text
RGB camera, 15-30 FPS
        |
        v
fixed feature extractors
  contour | track | HSV | MediaPipe | global illumination
        |
        v
independent RBF-SVM sensory specialists
  presence | motion | color | gesture | region quality | novelty
        |
        | margins + quality + temporal deltas
        v
fixed sparse recurrent BPU-shaped fusion core, T=4..8 steps
  sensory pool -> internal pool <recurrent> -> output pool
        |
        +----------------------+----------------------+
        |                      |                      |
 high confidence          ambiguous state         novel/open query
        |                      |                      |
 deterministic          Qwen text-only with      Qwen image path with
 sentence templates      shaped-state tokens      frame + shaped state
        |                      |                      |
        +----------------------+----------------------+
                               |
                               v
                  renderSpeech observation contract
```

### 1. Sensory specialists

The existing `train/insect-poc` specialists become modality-specific sensory
pools. Each produces more than a winning label:

```json
{
  "name": "motion",
  "class": "left",
  "scores": [-0.8, 1.4, -0.2, -0.6, -0.5],
  "margin": 1.6,
  "quality": 0.91,
  "age_frames": 12,
  "changed": true
}
```

The fusion input must retain decision margins, rejection state, track age, and
change signals. Passing only labels would discard the information needed for
calibrated routing.

Initial sensory groups:

| Pool | Inputs | Outputs |
|---|---|---|
| Presence | contour and persistence features | absent, present, uncertain |
| Motion | centroid/box history | still, left, right, up, down |
| Appearance | torso HSV histograms and crop quality | bounded color, unknown |
| Gesture | MediaPipe landmarks | six current gesture states, unknown |
| Environment | luminance and global pixel change | stable, light-change, camera-motion |
| Novelty | distance from known feature support | familiar, novel |

Each specialist remains independently replaceable and trainable with MLX. A
new appearance extractor must not require retraining gesture or motion.

### 2. Fixed recurrent shaped core

The first implementation should use a **connectome-shaped proxy**, not imply
biological fidelity prematurely:

- 512-1,024 recurrent units;
- signed sparse adjacency with 1-3% density;
- sensory, internal, and output blocks;
- fixed recurrent weights after initialization;
- spectral radius constrained below or near one for stable unrolling;
- four to eight recurrent propagation steps per camera update;
- recurrent state retained across frames and decayed when tracks disappear;
- trainable sensory projection `P_in` and output projection `P_out` only.

For state `h_t`, specialist vector `s_t`, fixed sparse adjacency `A`, and leak
factor `lambda`:

```text
h_(t,0) = lambda * h_(t-1,T) + P_in * s_t
h_(t,k+1) = tanh(A * h_(t,k) + h_(t,0))
o_t = P_out * h_(t,T)
```

`o_t` predicts:

- structured observation fields;
- confidence/calibration;
- whether the observation changed;
- route: `TEMPLATE`, `QWEN_TEXT`, or `QWEN_VISION`;
- optional safety veto.

After the proxy establishes a baseline, replace `A` with the paper's actual
connectome-derived adjacency only if the source data, preprocessing, polarity,
and licensing can be reproduced. The scientific comparison must include:

1. connectome adjacency;
2. degree/block-matched randomized adjacency;
3. fully random sparse reservoir;
4. no recurrence; and
5. a small trainable MLP/GRU with the same parameter budget.

Without these ablations, any improvement cannot be attributed to the biological
shape.

### 3. Qwen structured-state interface

There are two progressively stronger integrations.

#### Phase A: textual shaped-state conditioning

Do not send an image when the shaped core already captured the relevant state.
Send Qwen a compact record:

```text
SYSTEM: Convert trusted sensor state into one short observation. Never add an
attribute marked unknown.
STATE: person=present(0.97); motion=left(0.91); top=red(0.84);
gesture=unknown; illumination=stable; novelty=0.08
```

This still invokes the 2B language model, but skips visual encoding and reduces
the input to a stable, auditable vocabulary. It is useful when sentence variety
is desired but templates are insufficient.

#### Phase B: learned BPU prefix tokens

Project the shaped hidden state into a small sequence of Qwen-width embeddings:

```text
z = reshape(P_prefix * h, [K, 2048]), K=4..16
```

Insert `z` as learned prefix embeddings before the user instruction. Initially:

- freeze Qwen3.5-2B;
- freeze the recurrent adjacency;
- train `P_in`, `P_out`, and `P_prefix` on MLX;
- optionally add a small LoRA only after the frozen-Qwen baseline;
- supervise structured fields as well as final text.

This creates a real shaped-ensemble-to-language interface while keeping the
large model from relearning low-level motion/color rules. It does not make Qwen
itself biofidelic; the shaped core is an external recurrent visual adapter.

### 4. Three-route inference policy

#### Route 0: no semantic event

If the recurrent state and specialist outputs have not changed, emit nothing.
This should dominate a stationary camera stream.

#### Route 1: template fast path

Use deterministic text when all required fields clear their calibrated margins:

```text
Person detected, moving left, wearing red.
```

Target: at least 90-98% of frames and most ordinary semantic events.

#### Route 2: Qwen text path

Use text-only Qwen when fields are known but composition requires a less rigid
answer, conversational context, or user instruction following.

#### Route 3: Qwen vision path

Send the image only when:

- novelty exceeds its threshold;
- specialists disagree persistently;
- crop/track quality is inadequate;
- an open-vocabulary question cannot be answered from shaped state; or
- the user explicitly requests full visual inspection.

The route gate must be deterministic and logged. Qwen must not silently override
a safety-critical specialist without recording the disagreement.

## Training plan

### Stage 1: specialist calibration

Train current specialists independently with `train/insect-poc.sh`. Replace
synthetic contract data with grouped real feature datasets. Export margins and
validate by subject/environment rather than random frame.

### Stage 2: shaped-core training

Freeze the specialist models and recurrent adjacency. Train `P_in` and `P_out`
on sequences, not isolated frames. Labels include structured state, event
boundaries, and routing decisions. Optimize a combined loss:

```text
L = L_fields + 0.5*L_route + 0.25*L_change + 0.25*L_calibration
```

Class-weight the rare `QWEN_VISION` route, then tune its threshold against an
explicit energy/accuracy curve rather than maximizing raw routing accuracy.

### Stage 3: Qwen teacher distillation

Run Qwen3.5-2B offline over a diverse frame corpus and store:

- structured attributes;
- confidence/rejection flags;
- short observations;
- discrepancies with ground truth and specialists.

Human-review a stratified sample. Qwen output is noisy teacher data, not ground
truth. Train the shaped core to reproduce only bounded, verified fields.

### Stage 4: prefix adapter

Train the prefix projection against accepted Qwen observations. Compare:

- templates only;
- text serialization into frozen Qwen;
- learned prefix into frozen Qwen;
- prefix plus small LoRA;
- ordinary full-image Qwen.

Keep the prefix only if it improves grounded observation quality enough to
justify its training and runtime complexity.

### Stage 5: policy distillation loop

Every vision-path escalation becomes a candidate training example:

1. capture shaped state and selected frame;
2. obtain Qwen answer;
3. validate or correct the structured labels;
4. append to the responsible specialist/core dataset;
5. retrain only the affected specialist or projection;
6. confirm that the escalation rate falls on held-out recordings.

This turns Qwen into a temporary teacher for the edge system rather than a
permanent per-frame dependency.

## MLX implementation boundaries

On the M4 trainer:

- retain the current MLX RBF specialist backend;
- store fixed adjacency as a sparse index/value representation or benchmark a
  dense 512-1,024-unit matrix if MLX sparse operations are not advantageous;
- train projections and prefix adapter in FP16/BF16 with FP32 loss accumulation;
- perform sequence unrolling in MLX with truncated backpropagation;
- export specialist arrays and shaped-core arrays independently;
- keep Qwen fine-tuning optional and separate from core/specialist training.

On Raspberry Pi:

- run feature extraction, RBF specialists, recurrence, routing, and templates;
- use `uint8` images, quantized/fixed features, and FP32 or fixed-point
  accumulation initially;
- do not deploy Qwen3.5-2B unless a Pi benchmark demonstrates a useful role;
- treat Qwen as a remote/local-edge escalation service in the primary design.

## Artifact layout

```text
train/insect-poc/runs/
  specialists/
    presence/
    motion/
    color/
    gesture/
  shaped-core/
    adjacency.npz          # fixed, content-addressed
    input_projection.npz   # trainable
    output_projection.npz  # trainable
    routing.json
    meta.json
  qwen-adapter/
    prefix_projection.npz
    adapter-meta.json
  ensemble.json
```

Retraining rules:

- change one specialist: retrain that specialist, then optionally fine-tune
  `P_in` only;
- change routing thresholds: no model training;
- change recurrent adjacency: retrain both projections, never specialists;
- change Qwen version: retrain/evaluate the prefix projection, never the visual
  specialists by default;
- change sentence templates: no model training.

## Evaluation

The experiment succeeds only if it compares equal camera recordings and equal
semantic requirements.

### Quality

- structured field macro-F1;
- false presence events per hour;
- motion event F1 and detection delay;
- appearance accuracy with explicit unknown rate;
- unsupported-attribute/hallucination rate;
- temporal flicker and repeated-message rate;
- open-vocabulary answer quality on escalated frames;
- percentage of full-Qwen answers matched by the fast path.

### Efficiency

- camera-to-observation p50/p95 latency;
- average CPU/GPU utilization;
- resident memory;
- wall-plug joules per minute of identical video;
- joules per semantic event, not merely per classifier call;
- Qwen text calls per hour;
- Qwen vision calls per hour;
- percentage of frames producing no model work beyond feature extraction;
- thermal throttling and sustained FPS.

### Required ablation matrix

| System | Expected role |
|---|---|
| Qwen3.5-2B every sampled frame | quality/energy baseline |
| Current independent RBF ensemble | cheapest non-recurrent baseline |
| RBF + deterministic temporal rules | practical baseline |
| RBF + random fixed recurrent core | reservoir baseline |
| RBF + block-matched shaped core | topology hypothesis |
| RBF + connectome-derived core | biofidelity hypothesis |
| Shaped core + conditional Qwen | proposed complete system |

Report both total energy and semantic quality. A shaped model that saves energy
by producing fewer correct observations has not outcompeted the VLM.

## Initial success gates

On a fixed 60-minute replay suite:

1. at least 95% fewer Qwen vision invocations than the every-frame baseline;
2. at least 10x lower wall-plug energy for the complete observation pipeline;
3. p95 fast-path observation latency below 100 ms on Raspberry Pi;
4. no more than a five-point drop in bounded-field macro-F1;
5. unsupported-attribute rate lower than the Qwen baseline due to explicit
   rejection and templates;
6. at least 80% of routine semantic events handled without Qwen;
7. shaped-core topology must beat a degree/size-matched random reservoir across
   at least three seeds before attributing value to its shape.

## Principal risks

- The paper demonstrates task performance and parameter efficiency, not a live
  video energy advantage.
- The connectome result may not transfer to already-compressed specialist
  margins.
- Qwen teacher labels can encode hallucinations and dataset bias.
- Conditional routing can hide failures by escalating difficult examples.
- A recurrent core may add no value beyond simple hysteresis and a state
  machine; that is a valuable negative result and must remain in the ablations.
- Learned prefix tokens may be less reliable and less auditable than compact
  textual state.

## Recommended first experiment

Implement the smallest falsifiable slice before modifying Qwen weights:

1. add margin vectors and quality signals to each current specialist;
2. implement a 512-unit fixed sparse recurrent core in MLX;
3. train only input/output projections on recorded camera sequences;
4. compare it against deterministic rules and three random reservoirs;
5. route confident observations to templates;
6. route ambiguous observations to Qwen3.5-2B using textual shaped state;
7. measure full-system wall-plug energy and Qwen duty cycle;
8. proceed to prefix tokens only if recurrence improves calibrated sequence
   decisions and text-only Qwen is still a material bottleneck.

This sequence preserves the paper's strongest idea—fixed recurrent structure
with learned interfaces—while keeping the OpenPAVE experiment modular,
measurable, and honest about what the RBF ensemble can and cannot replace.
