# Detection Guide: Annotated Bounding Boxes for a Webcam Visualiser

How the reference template (`jepa/support/WORKING/17/viewer.py` + `yolo26.py`)
gets real-time, per-frame bounding boxes with text labels onto a live webcam
feed, drawn at the correct z-index (always on top, never covered by anything
else) — and how to build the equivalent using **Falcon Perception** instead of
YOLO26.

This is a companion to [dgx-spark-mlx-port.md](dgx-spark-mlx-port.md) §10.5–10.7,
which cover why OpenPAVE's own camera overlay does **not** currently use a
detector like this (a generic Haar-cascade attempt produced frequent false
positives and was pulled; OpenPAVE's `pave_ui/perception.py` instead asks the
loaded VLM itself to report `FEATURE: <label> <grid-cell>` lines). This guide
documents the two detector-based approaches as a reference for anyone who
wants a real per-frame detector in a future OpenPAVE iteration.

---

## Part 1 — How YOLO26 (MLX) drives the boxes today

Source: `jepa/support/WORKING/17/yolo26.py` + `viewer.py`.

### 1.1 The detector itself

`Yolo26Detector` (`yolo26.py:246-408`) wraps [`thewebAI/yolo-mlx`](https://github.com/thewebAI/yolo-mlx)
(`yolo26mlx.YOLO`) — a 100% MLX, NMS-free YOLO26 implementation. Its one public
inference entry point:

```python
# yolo26.py:358-398
def predict_square(self, square_bgr: np.ndarray) -> list[Detection]:
    """square_bgr: [S, S, 3] uint8 BGR camera centre-crop (unflipped).
    Returns detections in normalised [0,1] square coords."""
    ...
    px = self._predict_full(square_bgr)   # or sliced_predict() under SAHI
    ...
    out = []
    for (x1, y1, x2, y2, score, cid) in px:
        out.append(Detection(
            max(0.0, x1 / W), max(0.0, y1 / H),
            min(1.0, x2 / W), min(1.0, y2 / H),
            score, cid, self.names.get(int(cid), str(int(cid))),
        ))
    return out
```

`Detection` (`yolo26.py:234-243`) is a plain slotted struct:

```python
class Detection:
    """One detection in NORMALISED square coords [0,1] (x1<x2, y1<y2)."""
    __slots__ = ("x1", "y1", "x2", "y2", "score", "cls_id", "label")
```

Two details matter for everything that follows:
- Coordinates are **normalized [0,1]**, not pixels — the caller maps them to
  whatever display size it wants.
- They're computed against the **unflipped** camera crop. The webcam preview
  is mirrored for the user ("selfie view"), but the detector never sees the
  mirror — the mirror is applied only when converting to draw coordinates
  (§1.4). This keeps detector geometry stable regardless of display flipping.

### 1.2 Loading the model off the UI thread

```python
# viewer.py:271-299
class DetectorBuilder(QThread):
    progress = pyqtSignal(str)
    ready = pyqtSignal(str, object)
    failed = pyqtSignal(str)

    def __init__(self, name: str, use_sahi: bool, slice_size: int, max_tiles: int):
        super().__init__()
        self.name = name
        ...

    def run(self):
        try:
            if self.name == "Off":
                self.ready.emit(self.name, None)
                return
            self.progress.emit(f"Loading detector {self.name}")
            from yolo26 import Yolo26Detector
            det = Yolo26Detector(variant=self.name, use_sahi=self.use_sahi, ...)
            self.ready.emit(self.name, det)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
```

Selecting a detector from the dropdown (`_select_detector`, `viewer.py:1248-1275`)
spins up one `DetectorBuilder`, and `_on_detector_built`/`_attach_detector`
(`viewer.py:1221-1246,1277-1288`) store the resulting `Yolo26Detector` instance
and build a `DetectorWorker` around it once it's ready. Loading (weight I/O,
first MLX compile) never blocks the UI thread.

### 1.3 Running inference every tick, still off the UI thread

```python
# viewer.py:545-570
class DetectorWorker(QThread):
    detections_ready = pyqtSignal(object, float)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.square = None

    def submit(self, square_bgr):
        if self.isRunning():      # busy -> caller just skips this tick
            return False
        self.square = square_bgr
        return True

    def run(self):
        if self.square is None or self.detector is None:
            return
        t0 = time.perf_counter()
        dets = self.detector.predict_square(self.square)
        self.detections_ready.emit(dets, time.perf_counter() - t0)
```

A `QTimer` drives the cadence — **150ms** normally, **450ms** under SAHI tiling
(`_detector_interval_ms`, `viewer.py:1193-1195`):

```python
# viewer.py:1683-1711 (abridged)
def _maybe_detect(self):
    if self.detector is None or self.det_worker is None:
        return
    if self.det_worker.isRunning():          # one job in flight at a time
        return
    square = self._current_square_crop()      # center-crop, UNFLIPPED
    if square is None:
        return
    if self.det_worker.submit(square):
        self.det_worker.start()

def _on_detections_ready(self, detections, dt):
    self.detections = list(detections or [])  # picked up by the next _render()
```

Note `DetectorWorker.submit()` returns `False` (and the caller just does
nothing that tick) if a detection job is already running — one `DetectorWorker`
instance is reused for the detector's entire lifetime; a new `QThread` is never
spun up per frame.

### 1.4 The z-index: why boxes are always on top

This is the part worth reading closely — it's a two-pass paint, and the boxes
are drawn in the **second, later** pass:

```python
# viewer.py:1877-1917 (abridged)
def _render(self):
    ...
    # 1) base camera frame: mirrored, cropped to a centre square
    rgb = np.ascontiguousarray(cv2.cvtColor(cv2.flip(square, 1), cv2.COLOR_BGR2RGB))
    pixmap = QPixmap.fromImage(QImage(rgb.data, s, s, s * 3, QImage.Format.Format_RGB888))

    # 2) feature-heatmap overlay, painted onto the RAW-resolution pixmap
    painter = QPainter(pixmap)
    if self.overlay_rgba is not None:
        ov_img = QImage(ov.data, ov.shape[1], ov.shape[0], ov.shape[1] * 4,
                        QImage.Format.Format_RGBA8888).copy()
        painter.drawImage(QRect(0, 0, s, s), ov_img)
    painter.end()                              # <- this paint pass is DONE

    # 3) scale the (frame + heatmap) composite to the display widget's size
    final_pm = pixmap.scaled(self.display.size(), Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.FastTransformation)

    # 4) detection boxes: a SEPARATE, LATER QPainter pass, on the ALREADY-SCALED pixmap
    if self.detections:
        draw_s = min(final_pm.width(), final_pm.height())
        det_p = QPainter(final_pm)
        self._draw_detections(det_p, draw_s)
        det_p.end()

    self.display.setPixmap(final_pm)
```

The z-order guarantee comes from **paint pass ordering, not a z-index
property**: camera frame and heatmap are composited and scaled *first*, in
their own `QPainter` that is explicitly `.end()`-ed; detection boxes are then
painted in a *second* `QPainter` opened on that already-finished pixmap. Since
Qt paints in call order within a single `QPainter`, and this uses two
sequential painters on the same pixmap, boxes can never end up underneath the
heatmap or blurred by the scaling step applied to the base image — they're
literally the last pixels written before the widget shows the frame.

*(This is also the exact bug OpenPAVE hit and fixed in
[dgx-spark-mlx-port.md §10.5](dgx-spark-mlx-port.md): drawing a full-width
status band **after** a box painted over its label. The fix there was the same
principle — draw the thing that must stay on top **last**.)*

### 1.5 Drawing one box + label

```python
# viewer.py:1819-1858 (abridged)
def _draw_detections(self, painter: QPainter, s: int):
    box_pen = QPen(QColor(*DETECTION_LABEL_BACKGROUND)); box_pen.setWidth(2)
    text_pen = QPen(QColor(*DETECTION_LABEL_FOREGROUND))
    label_bg = QColor(*DETECTION_LABEL_BACKGROUND)
    font = QFont("Menlo"); font.setPixelSize(max(12, int(DETECTION_LABEL_PX)))
    painter.setFont(font)

    for det in self.detections:
        # detector coords are in the UNFLIPPED square; display IS mirrored ->
        # mirror x here, once, at draw time (never in the detector itself)
        px1 = int(round(s - det.x2 * s)); px2 = int(round(s - det.x1 * s))
        py1 = int(round(det.y1 * s));     py2 = int(round(det.y2 * s))
        x1, x2 = max(0, min(px1, px2, s-1)), max(0, min(max(px1, px2), s))
        y1, y2 = max(0, min(py1, py2, s-1)), max(0, min(max(py1, py2), s))

        painter.setPen(box_pen)
        painter.drawRect(x1, y1, max(1, x2-x1), max(1, y2-y1))

        label = f"{det.label} {det.score:.2f}"
        text_rect = QRect(x1 + 1, y1 + 1, max(44, len(label)*9 + 8), max(14, DETECTION_LABEL_PX + 4))
        painter.fillRect(text_rect, label_bg)      # label chip INSIDE the box's corner
        painter.setPen(text_pen)
        painter.drawText(x1 + 3, y1 + 1 + text_rect.height() - 4, label)
```

Constants: `DETECTION_LABEL_PX = 14`, `DETECTION_LABEL_BACKGROUND = (255,0,0,32)`
(translucent red — used for both the box outline *and* the label chip fill),
`DETECTION_LABEL_FOREGROUND = (255,255,255)` (`viewer.py:110-112`).

The label chip is deliberately placed **inside** the box's own top-left corner
(`text_rect` starts at `x1+1, y1+1`, not above/outside the box), so a box near
the edge of the frame never has its label clipped off-screen.

---

## Part 2 — Porting this to Falcon Perception instead of YOLO26

Source explored: `/Users/scottphillips/Documents/GitHub/Falcon-Perception`.

### 2.1 What's actually different

| | YOLO26 (MLX) | Falcon Perception (MLX) |
|---|---|---|
| Model type | Single dense forward pass detector | Autoregressive, natively multimodal Transformer |
| Vocabulary | Fixed — whatever classes the weights were trained/converted with (COCO-80 for stock variants) | **Open-vocabulary** — any natural-language query, e.g. `"hand"`, `"face and hand"`, `"the person on the left"` |
| Output | Boxes only | Boxes **and/or pixel-accurate segmentation masks** (task-selectable) |
| Box format | `x1,y1,x2,y2` normalized [0,1] | `{x,y}` (normalized **center**) + `{h,w}` (normalized height/width) — different convention, must convert |
| Latency shape | One forward pass — fast, ~consistent | One token per detected coordinate, decoded autoregressively (`max_new_tokens`, default 200 in the demo) — slower, and scales with how much it finds |
| Repo dependency | `yolo26mlx` only | `pip install -e ".[mlx]"` (MLX backend has **no PyTorch/transformers dependency** at inference time) |

Because of the latency shape difference, treat this as a **slower-cadence**
detector than YOLO26 in the same architecture (§2.5), not a drop-in doubling of
tick rate — measure actual per-call latency on your hardware before picking a
timer interval.

### 2.2 Installing and loading the MLX model

```bash
# From the Falcon-Perception repo
pip install -e ".[mlx]"
```

```python
# Falcon-Perception/falcon_perception/__init__.py:363-391
from falcon_perception import PERCEPTION_MODEL_ID, load_and_prepare_model

model, tokenizer, model_args = load_and_prepare_model(
    hf_model_id=PERCEPTION_MODEL_ID,   # "tiiuae/Falcon-Perception"
                                        # or PERCEPTION_300M_MODEL_ID for the smaller variant
    dtype="float16",
    backend="mlx",
)
```

This auto-downloads the HF export (`snapshot_download`) and converts
safetensors → MLX weights on the fly
(`falcon_perception/mlx/convert.py`, called from `load_from_hf_export_mlx`,
`__init__.py:309-360`) — no local `.pt`→`.npz` conversion step to script
yourself, unlike YOLO26's `vjepa21.sh` pipeline.

Same "load off the UI thread" pattern as `DetectorBuilder`:

```python
class FalconDetectorBuilder(QThread):
    progress = pyqtSignal(str)
    ready = pyqtSignal(object, object, object)   # model, tokenizer, model_args
    failed = pyqtSignal(str)

    def run(self):
        try:
            self.progress.emit("Loading Falcon Perception (MLX)")
            from falcon_perception import PERCEPTION_300M_MODEL_ID, load_and_prepare_model
            model, tok, args = load_and_prepare_model(
                hf_model_id=PERCEPTION_300M_MODEL_ID,  # smaller/faster for a live overlay
                dtype="float16", backend="mlx",
            )
            self.ready.emit(model, tok, args)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
```

### 2.3 Building the query and running inference

```python
# Falcon-Perception/falcon_perception/__init__.py:296-306
def build_prompt_for_task(query: str, task: str) -> str:
    if task in ("segmentation", "detection"):
        prefix = "Segment these expressions in the image:"
        return f"<|image|>{prefix}<|start_of_query|>{query}<|REF_SEG|>"
    ...
```

```python
# Falcon-Perception/demo/perception_single_mlx.py:167-207 (abridged)
from falcon_perception.mlx.batch_inference import BatchInferenceEngine, process_batch_and_generate

engine = BatchInferenceEngine(model, tokenizer)
prompt = build_prompt_for_task("hand", "detection")   # your query, "detection"-only skips masks

batch = process_batch_and_generate(
    tokenizer, [(pil_image, prompt)],
    max_length=model_args.max_seq_len, min_dimension=256, max_dimension=1024,
)
output_tokens, aux_outputs = engine.generate(
    tokens=batch["tokens"], pos_t=batch["pos_t"], pos_hw=batch["pos_hw"],
    pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
    max_new_tokens=80, temperature=0.0, task="detection",
)

aux = aux_outputs[0]
bboxes = pair_bbox_entries(aux.bboxes_raw)   # -> [{x, y, h, w}, ...] normalized center+size
```

`pair_bbox_entries` (`demo/perception_single_mlx.py:35-45`) is a ~6-line
function — copy it, it has no heavy dependency:

```python
def pair_bbox_entries(raw: list[dict]) -> list[dict]:
    """Pair [{x,y}, {h,w}, ...] into [{x,y,h,w}, ...]."""
    bboxes, current = [], {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        current.update(entry)
        if all(k in current for k in ("x", "y", "h", "w")):
            bboxes.append(dict(current))
            current = {}
    return bboxes
```

For a live webcam frame instead of a file, skip `load_image`/`Path` handling
and hand a `PIL.Image` built straight from the current BGR frame:
`Image.fromarray(frame_bgr[:, :, ::-1])`.

### 2.4 Converting to the SAME `Detection` shape — reuse `_draw_detections` as-is

This is the key integration move: don't write a second drawing routine. Adapt
Falcon's `{x,y,h,w}` (center + size) into the exact `Detection` shape YOLO26
already produces (`x1,y1,x2,y2` normalized), and `_draw_detections`/the z-index
pipeline in Part 1 needs **zero changes**.

```python
from yolo26 import Detection   # reuse the existing slotted struct

def falcon_bboxes_to_detections(bboxes: list[dict], label_query: str) -> list[Detection]:
    """{x,y,h,w} normalized center+size -> Detection(x1,y1,x2,y2) normalized corners."""
    out = []
    for b in bboxes:
        x1 = b["x"] - b["w"] / 2.0
        y1 = b["y"] - b["h"] / 2.0
        x2 = b["x"] + b["w"] / 2.0
        y2 = b["y"] + b["h"] / 2.0
        out.append(Detection(
            max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2),
            1.0, 0, label_query,   # no per-box confidence/class id from this model — use the query text as the label
        ))
    return out
```

(Falcon Perception doesn't emit a numeric confidence or a class id per box the
way YOLO26 does — every box it returns is one it decided actually matches the
query, so `score` is set to `1.0` and `label_query` — the text you asked for —
becomes the label. If you query multiple things at once, e.g.
`"face, hand"`, you'll need your own convention for mapping boxes back to
which term matched, since the model returns one flat list of boxes for the
whole query.)

### 2.5 Wiring it into the same QThread/timer/z-index pipeline

```python
class FalconDetectorWorker(QThread):
    detections_ready = pyqtSignal(object, float)

    def __init__(self, model, tokenizer, model_args, query: str):
        super().__init__()
        self.model, self.tokenizer, self.model_args = model, tokenizer, model_args
        self.query = query
        self.frame_bgr = None

    def submit(self, frame_bgr) -> bool:
        if self.isRunning():          # exactly like DetectorWorker.submit()
            return False
        self.frame_bgr = frame_bgr
        return True

    def run(self):
        if self.frame_bgr is None:
            return
        from PIL import Image
        from falcon_perception import build_prompt_for_task
        from falcon_perception.mlx.batch_inference import BatchInferenceEngine, process_batch_and_generate

        t0 = time.perf_counter()
        pil_image = Image.fromarray(self.frame_bgr[:, :, ::-1])
        prompt = build_prompt_for_task(self.query, "detection")
        engine = BatchInferenceEngine(self.model, self.tokenizer)   # cheap: no weights copied
        batch = process_batch_and_generate(
            self.tokenizer, [(pil_image, prompt)],
            max_length=self.model_args.max_seq_len, min_dimension=256, max_dimension=768,
        )
        _, aux_outputs = engine.generate(
            tokens=batch["tokens"], pos_t=batch["pos_t"], pos_hw=batch["pos_hw"],
            pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
            max_new_tokens=60, temperature=0.0, task="detection",
        )
        bboxes = pair_bbox_entries(aux_outputs[0].bboxes_raw)
        dets = falcon_bboxes_to_detections(bboxes, self.query)
        self.detections_ready.emit(dets, time.perf_counter() - t0)
```

Wire it exactly like §1.3/§1.4: a `QTimer` calls a `_maybe_detect`-style method
that checks `not self.falcon_worker.isRunning()` before `submit()`+`start()`;
`_on_detections_ready` sets `self.detections = list(dets)`; `_render()`'s
Part-1.4 two-pass paint (heatmap first, `_draw_detections(painter, draw_s)`
**last**) is untouched — it only ever consumed `self.detections`, and every
element is now a `Detection`, regardless of which backend produced it.

**Timer interval**: don't reuse YOLO26's 150ms. Benchmark
`FalconDetectorWorker.run()`'s wall time on your hardware first — autoregressive
decode over `max_new_tokens=60` will be materially slower than one YOLO26
forward pass — and set the interval a comfortable margin above the measured
p95, the same way `_detector_interval_ms()` picks 450ms specifically for
SAHI's heavier tiled inference (§1.3).

### 2.6 Before you commit to this for OpenPAVE specifically

- **Torch-free is required, not optional, here.** `falcon_perception/visualization_utils.py`
  hard-imports `torch`/`torchvision` — do not import it for the MLX path; the
  `Detection`-conversion + reused `_draw_detections` approach above avoids that
  entirely, matching the README's own claim that the MLX backend needs no
  PyTorch/transformers at inference time.
- **This is a real detector, unlike OpenPAVE's current approach.** OpenPAVE's
  `pave_ui/perception.py` (see [dgx-spark-mlx-port.md §10.7](dgx-spark-mlx-port.md))
  asks the *already-loaded* VLM (Gemma/Qwen) to self-report a `FEATURE: <label>
  <grid-cell>` line in the same call that produces the STOP/TROT/etc. intent —
  zero extra model, zero extra latency, but coarse (9-cell grid, not a real
  box). Falcon Perception would give real, tight boxes and true open-vocabulary
  queries, at the cost of loading and running a **second** model alongside
  whichever VLM drives intent, and adding its own (likely multi-hundred-ms)
  latency per query.
- **Pick model size deliberately.** `PERCEPTION_300M_MODEL_ID` over the full
  `PERCEPTION_MODEL_ID` is the more realistic starting point for a live,
  on-device overlay on a MacBook, mirroring the "prefer the lightweight model"
  guidance already established for the VLM side in
  [dgx-spark-mlx-port.md §10.2](dgx-spark-mlx-port.md).
