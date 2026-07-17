"""Train/evaluate the standalone MediaPipe-compatible 21-point landmarker tower.

Runtime contract: 96x96 RGB -> landmarks[42], visibility[21], presence[1], quality[1].
MediaPipe is a training teacher only; the exported ONNX and GUI runtime do not import it.

Run:
  .venv/bin/python train/landmark_tower.py train
  .venv/bin/python train/landmark_tower.py eval
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

TRAIN = Path(__file__).resolve().parent
DATASETS = TRAIN / "datasets"
OUT = TRAIN / "runs" / "landmark_tower"
GESTURE = TRAIN / "runs" / "tiny_gesture"
SOURCES = ("jester", "hagrid_shapes", "hagrid", "crude", "yolo26", "ipn", "swipe_phases")
INPUT = 128
GRID = 32


def _load(sources, fraction=1.0):
    imgs = []; lms = []; has = []; known = []; val = []; used = []
    for source in sources:
        path = DATASETS / source / "prepared.npz"
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=True)
        if "landmarks" not in d.files:
            continue
        sel = np.arange(len(d["imgs"]))
        if fraction < 1:
            tr = sel[~d["is_val"]]; va = sel[d["is_val"]]
            stride = max(1, round(1 / fraction)); sel = np.sort(np.r_[tr[::stride], va])
        imgs.append(d["imgs"][sel]); lms.append(d["landmarks"][sel])
        has.append(d["has_lm"][sel]); val.append(d["is_val"][sel]); used.append(source)
        if "presence_known" in d.files:
            known.append(d["presence_known"][sel])
        else:
            # Hand-centric datasets: a teacher miss is unknown, not absence.
            # Only crude's explicit no_hand frames are trusted negatives.
            labels = d["labels"][sel] if "labels" in d.files else np.full(len(sel), "")
            known.append(d["has_lm"][sel] | ((source == "crude") & (labels == "no_hand")))
    if not imgs:
        raise RuntimeError("no prepared v2 landmark shards; run ./train/gesture-lab.sh prepare")
    return tuple(map(np.concatenate, (imgs, lms, has, known, val))) + (used,)


def _model():
    import torch
    import torch.nn as nn

    class DS(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
                                     nn.BatchNorm2d(cin), nn.SiLU(),
                                     nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU())
        def forward(self, x): return self.net(x)

    class LandmarkTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU(),
                                     DS(32, 48, 2), DS(48, 80), DS(80, 112, 2), DS(112, 160),
                                     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     nn.Conv2d(160, 96, 1), nn.SiLU())
            self.heat = nn.Conv2d(96, 21, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.pres = nn.Linear(96, 1); self.qual = nn.Linear(96, 1)
            yy, xx = torch.meshgrid(torch.linspace(0, 1, GRID), torch.linspace(0, 1, GRID), indexing="ij")
            self.register_buffer("xx", xx.reshape(-1)); self.register_buffer("yy", yy.reshape(-1))

        def forward(self, x):
            f = self.enc(x); heat = self.heat(f)
            prob = heat.flatten(2).softmax(-1)
            px = (prob * self.xx).sum(-1); py = (prob * self.yy).sum(-1)
            lm = torch.stack((px, py), -1).flatten(1)
            visibility = prob.amax(-1) * (GRID * GRID)
            pooled = self.pool(f).flatten(1)
            return lm, visibility, self.pres(pooled), self.qual(pooled), heat
    return LandmarkTower()


def _detector_model():
    import torch
    import torch.nn as nn

    class DS(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
                                     nn.BatchNorm2d(cin), nn.SiLU(),
                                     nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU())
        def forward(self, x): return self.net(x)

    class HandDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv2d(3, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU(),
                                     DS(24, 40, 2), DS(40, 64), DS(64, 80, 2),
                                     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     nn.Conv2d(80, 64, 1), nn.SiLU())
            self.centre_heat = nn.Conv2d(64, 1, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.side = nn.Linear(64, 1); self.presence = nn.Linear(64, 1)
            yy, xx = torch.meshgrid(torch.linspace(0, 1, GRID), torch.linspace(0, 1, GRID), indexing="ij")
            self.register_buffer("xx", xx.reshape(-1)); self.register_buffer("yy", yy.reshape(-1))

        def forward(self, x):
            f = self.enc(x); heat = self.centre_heat(f); prob = heat.flatten(2).softmax(-1)
            cx = (prob[:, 0] * self.xx).sum(-1); cy = (prob[:, 0] * self.yy).sum(-1)
            pooled = self.pool(f).flatten(1)
            return torch.stack((cx, cy), -1), self.side(pooled).sigmoid(), self.presence(pooled), heat
    return HandDetector()


def _heatmaps(lm, valid):
    import torch
    xy = lm.view(-1, 21, 2).clamp(0, 1)
    axis = torch.linspace(0, 1, GRID, device=lm.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dist = (xx[None, None] - xy[:, :, 0, None, None]) ** 2 + (yy[None, None] - xy[:, :, 1, None, None]) ** 2
    target = torch.exp(-dist / (2 * (1.4 / GRID) ** 2))
    # A probability distribution per landmark. Sparse MSE is dominated by the
    # 575 background cells and admits an all-background collapse; normalized
    # spatial cross-entropy below cannot win without locating the joint.
    return target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def _centre_heatmaps(points):
    import torch
    axis = torch.linspace(0, 1, GRID, device=points.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dist = ((xx[None] - points[:, 0, None, None]) ** 2
            + (yy[None] - points[:, 1, None, None]) ** 2)
    target = torch.exp(-dist / (2 * (1.5 / GRID) ** 2))
    return target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def _hand_crops(x, target, valid):
    """Canonical jittered hand crops and crop-local landmark targets."""
    import torch
    import torch.nn.functional as F
    rows = torch.nonzero(valid, as_tuple=False).flatten()
    pts = target[rows].reshape(-1, 21, 2)
    lo, hi = pts.amin(1), pts.amax(1)
    centre = (lo + hi) * .5
    bbox_side = (hi - lo).amax(1).clamp(.025, .8)
    side = (bbox_side * (1.65 + .55 * torch.rand(len(rows), device=x.device))).clamp(.10, 1.15)
    centre = centre + torch.randn_like(centre) * side[:, None] * .06
    offset = centre - side[:, None] * .5
    theta = torch.zeros(len(rows), 2, 3, device=x.device)
    theta[:, 0, 0] = side; theta[:, 1, 1] = side
    theta[:, 0, 2] = side + 2 * offset[:, 0] - 1
    theta[:, 1, 2] = side + 2 * offset[:, 1] - 1
    grid = F.affine_grid(theta, (len(rows), x.shape[1], INPUT, INPUT), align_corners=False)
    crops = F.grid_sample(x[rows], grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    local = ((pts - offset[:, None]) / side[:, None, None]).clamp(0, 1)
    flip = torch.rand(len(rows), device=x.device) < .5
    if flip.any():
        crops[flip] = crops[flip].flip(-1); local[flip, :, 0] = 1 - local[flip, :, 0]
    return rows, crops, local.flatten(1)


def _spatial_augment(x, target, valid):
    """Landmark-aware camera/crop augmentation on device.

    Each positive hand is rescaled to occupy 15-70% of the output and moved
    around the frame. This simulates loose body crops, close webcam hands and
    off-centre viewpoints while updating the coordinate targets exactly.
    """
    import torch
    import torch.nn.functional as F

    n = len(x)
    theta = torch.eye(2, 3, device=x.device)[None].repeat(n, 1, 1)
    out_target = target.clone()
    rows = torch.nonzero(valid, as_tuple=False).flatten()
    if len(rows):
        pts = target[rows].reshape(-1, 21, 2)
        lo, hi = pts.amin(1), pts.amax(1)
        centre = (lo + hi) * .5
        side = (hi - lo).amax(1).clamp(.025, .95)
        desired = .15 + .55 * torch.rand(len(rows), device=x.device)
        sample_scale = (side / desired).clamp(.18, 1.20)
        out_centre = .32 + .36 * torch.rand(len(rows), 2, device=x.device)
        offset01 = centre - sample_scale[:, None] * out_centre
        theta[rows, 0, 0] = sample_scale
        theta[rows, 1, 1] = sample_scale
        # affine_grid works in [-1,1]: src = scale*out + (scale+2*offset-1)
        theta[rows, 0, 2] = sample_scale + 2 * offset01[:, 0] - 1
        theta[rows, 1, 2] = sample_scale + 2 * offset01[:, 1] - 1
        moved = (pts - offset01[:, None]) / sample_scale[:, None, None]
        out_target[rows] = moved.flatten(1).clamp(0, 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    flip = torch.rand(n, device=x.device) < .5
    if flip.any():
        x[flip] = x[flip].flip(-1)
        flipped = out_target[flip].reshape(-1, 21, 2)
        flipped[:, :, 0] = 1 - flipped[:, :, 0]
        out_target[flip] = flipped.flatten(1)
    return x, out_target


def train(args):
    import cv2
    import torch
    import torch.nn.functional as F

    imgs, lms, has, known, val, used = _load(args.sources, args.data_fraction)
    device = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    # Resize once; uint8 storage avoids a multi-GB float tensor.
    resized = np.stack([cv2.resize(x, (INPUT, INPUT), interpolation=cv2.INTER_AREA) for x in imgs])
    model = _model().to(device); detector = _detector_model().to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(detector.parameters()), 5e-4, weight_decay=1e-4)
    tr = np.flatnonzero(~val); va = np.flatnonzero(val)
    best = None; best_score = -1.; stale = 0; started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); detector.train(); np.random.shuffle(tr)
        for rows in np.array_split(tr, max(1, len(tr) // args.batch)):
            x = (torch.tensor(resized[rows], device=device).permute(0, 3, 1, 2)
                 .contiguous().float() / 255 - .5)
            target = torch.tensor(lms[rows], device=device).float()
            valid = torch.tensor(has[rows], device=device).bool()
            presence_known = torch.tensor(known[rows], device=device).bool()
            x, target = _spatial_augment(x, target, valid)
            # Lighting, sensor-noise and contrast diversity.
            x = x * (.75 + .5 * torch.rand(len(rows), 1, 1, 1, device=device)) + torch.randn_like(x) * .025
            centre, side_pred, presence, centre_heat = detector(x)
            loss = torch.zeros((), device=device)
            if presence_known.any():
                loss = F.binary_cross_entropy_with_logits(
                    presence[presence_known, 0], valid[presence_known].float())
            if valid.any():
                pts = target[valid].reshape(-1, 21, 2)
                true_centre = (pts.amin(1) + pts.amax(1)) * .5
                true_side = ((pts.amax(1) - pts.amin(1)).amax(1) * 2.0).clamp(.08, 1.2)
                loss = loss + 5 * F.smooth_l1_loss(centre[valid], true_centre)
                loss = loss + 2 * F.smooth_l1_loss(side_pred[valid, 0], true_side)
                ch = _centre_heatmaps(true_centre).flatten(1)
                loss = loss - (ch * F.log_softmax(centre_heat[valid].flatten(1), -1)).sum(-1).mean()

                _crop_rows, crops, crop_target = _hand_crops(x, target, valid)
                pred, _vis, crop_presence, quality, heat = model(crops)
                loss = loss + F.binary_cross_entropy_with_logits(
                    crop_presence[:, 0], torch.ones(len(crops), device=device))
                loss = loss + 10 * F.smooth_l1_loss(pred, crop_target)
                target_heat = _heatmaps(crop_target, valid[valid]).flatten(2)
                heat_logp = F.log_softmax(heat.flatten(2), dim=-1)
                loss = loss - (target_heat * heat_logp).sum(-1).mean()
                err = (pred.reshape(-1, 21, 2) - crop_target.reshape(-1, 21, 2)).norm(dim=2).mean(1)
                loss = loss + F.smooth_l1_loss(quality[:, 0].sigmoid(), (1 - err * 5).clamp(0, 1))
            opt.zero_grad(); loss.backward()
            # MPS depthwise-convolution gradients can be non-contiguous. The
            # default foreach implementation calls view() and crashes; the
            # scalar path is equivalent and works on MPS, CPU, and CUDA.
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(detector.parameters()),
                                           2, foreach=False)
            opt.step()
        metrics = evaluate_model(detector, model, resized[va], lms[va], has[va], known[va], device)
        # Landmark geometry is the tower's purpose; presence alone must never
        # select a geometrically collapsed checkpoint.
        score = 4 * metrics["pck_010"] + metrics["pck_005"] + .25 * metrics["presence_f1"]
        print(f"[landmarks] epoch {epoch+1}: pck.10={metrics['pck_010']:.3f} presence={metrics['presence_f1']:.3f}")
        if score > best_score:
            best_score = score; stale = 0
            best = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    {k: v.detach().cpu().clone() for k, v in detector.state_dict().items()})
        else: stale += 1
        if stale >= args.patience: break
    model.load_state_dict(best[0]); detector.load_state_dict(best[1]); OUT.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval(); detector = detector.cpu().eval()

    class Export(torch.nn.Module):
        def __init__(self, inner): super().__init__(); self.inner = inner
        def forward(self, x):
            lm, vis, pres, quality, _ = self.inner(x)
            return lm, vis, pres.sigmoid(), quality.sigmoid()
    torch.onnx.export(Export(model), torch.zeros(1, 3, INPUT, INPUT), OUT / "model.onnx",
                      input_names=["image"], output_names=["landmarks", "visibility", "presence", "quality"],
                      dynamic_axes={"image": {0: "n"}, "landmarks": {0: "n"}, "visibility": {0: "n"},
                                    "presence": {0: "n"}, "quality": {0: "n"}}, dynamo=False)
    class ExportDetector(torch.nn.Module):
        def __init__(self, inner): super().__init__(); self.inner = inner
        def forward(self, x):
            centre, side, presence, _ = self.inner(x)
            return centre, side, presence.sigmoid()
    torch.onnx.export(ExportDetector(detector), torch.zeros(1, 3, INPUT, INPUT), OUT / "detector.onnx",
                      input_names=["image"], output_names=["centre", "size", "presence"],
                      dynamic_axes={"image": {0: "n"}, "centre": {0: "n"},
                                    "size": {0: "n"}, "presence": {0: "n"}}, dynamo=False)
    final = evaluate_onnx(resized[va], lms[va], has[va], known[va])
    accepted = (final["presence_f1"] >= .90 and final["pck_010"] >= .50
                and final["nme"] <= .35)
    meta = {"contract": "openpave.landmarker.v2", "input_px": INPUT,
            "architecture": "hand-detector -> canonical-crop -> 21-point-landmarker",
            "landmark_order": "mediapipe-21-compatible", "coordinate_space": "frame-normalized",
            "params": sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in detector.parameters()),
            "precision": "fp32",
            "sources": used, "epochs": epoch + 1, "seconds": time.perf_counter() - started,
            "accepted": accepted,
            "acceptance": {"presence_f1_min": .90, "pck_010_min": .50, "nme_max": .35},
            "metrics": final}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[landmarks] saved {OUT/'model.onnx'}: {final}")


def _torch_pipeline(detector, model, x):
    import torch.nn.functional as F
    centre, side, presence, _ = detector(x)
    side = side[:, 0].clamp(.08, 1.2)
    offset = centre - side[:, None] * .5
    theta = x.new_zeros(len(x), 2, 3)
    theta[:, 0, 0] = side; theta[:, 1, 1] = side
    theta[:, 0, 2] = side + 2 * offset[:, 0] - 1
    theta[:, 1, 2] = side + 2 * offset[:, 1] - 1
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    crops = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    local, _, _, quality, _ = model(crops)
    full = centre[:, None] + (local.reshape(-1, 21, 2) - .5) * side[:, None, None]
    return full.flatten(1), presence[:, 0].sigmoid(), quality[:, 0].sigmoid()


def evaluate_model(detector, model, imgs, lms, has, known, device):
    import torch
    preds=[]; presence=[]
    model.eval(); detector.eval()
    with torch.no_grad():
        for rows in np.array_split(np.arange(len(imgs)), max(1, len(imgs)//256)):
            x=(torch.tensor(imgs[rows],device=device).permute(0,3,1,2).contiguous().float()/255-.5)
            lm,p,_q=_torch_pipeline(detector,model,x)
            preds.append(lm.cpu().numpy()); presence.append(p.cpu().numpy())
    return landmark_metrics(np.concatenate(preds), np.concatenate(presence), lms, has, known)


def landmark_metrics(pred, presence, target, has, known=None):
    from sklearn.metrics import f1_score
    positive = has.astype(bool); errors=np.zeros((0,21),np.float32)
    if positive.any():
        p=pred[positive].reshape(-1,21,2); t=target[positive].reshape(-1,21,2)
        palm=np.linalg.norm(t[:,9]-t[:,0],axis=1).clip(.03)[:,None]
        errors=np.linalg.norm(p-t,axis=2)/palm
    known = np.ones(len(has), bool) if known is None else np.asarray(known, bool)
    return {"presence_f1": float(f1_score(positive[known], presence[known]>=.5,zero_division=0)),
            "nme": float(errors.mean()) if errors.size else 999.,
            "pck_005": float((errors<=.05).mean()) if errors.size else 0.,
            "pck_010": float((errors<=.10).mean()) if errors.size else 0.}


def evaluate_onnx(imgs,lms,has,known=None):
    runtime = LandmarkerRuntime()
    pred=[]; pres=[]; times=[]
    for im in imgs:
        t=time.perf_counter(); lm,p,_q=runtime.step(im, apply_gate=False)
        times.append((time.perf_counter()-t)*1000); pred.append(lm); pres.append(p)
    m=landmark_metrics(np.asarray(pred),np.asarray(pres),lms,has,known)
    m["median_ms"]=float(np.median(times)); return m


class LandmarkerRuntime:
    """Two-stage CPU ONNX landmarker: full frame -> ROI -> 21 points."""
    def __init__(self, presence_gate=.5, quality_gate=.15):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(
            1, int(os.environ.get("PAVE_ORT_THREADS", "4")))
        options.inter_op_num_threads = 1
        self.detector = ort.InferenceSession(
            str(OUT/"detector.onnx"), sess_options=options,
            providers=["CPUExecutionProvider"])
        self.landmarker = ort.InferenceSession(
            str(OUT/"model.onnx"), sess_options=options,
            providers=["CPUExecutionProvider"])
        self.presence_gate=presence_gate; self.quality_gate=quality_gate

    def step(self, rgb, apply_gate=True):
        import cv2
        frame=cv2.resize(rgb,(INPUT,INPUT))
        x=(frame.astype(np.float32)/255-.5).transpose(2,0,1)[None]
        centre,size,presence=self.detector.run(None,{"image":x})
        centre=centre[0]; side=float(np.clip(size[0,0],.08,1.2)); p=float(presence[0,0])
        side_px=max(8,int(round(side*INPUT)))
        crop=cv2.getRectSubPix(frame,(side_px,side_px),(float(centre[0]*INPUT),float(centre[1]*INPUT)))
        xc=(cv2.resize(crop,(INPUT,INPUT)).astype(np.float32)/255-.5).transpose(2,0,1)[None]
        local,_vis,_crop_presence,quality=self.landmarker.run(None,{"image":xc})
        q=float(quality[0,0]); full=centre[None]+(local[0].reshape(21,2)-.5)*side
        lm=np.clip(full,0,1).astype(np.float32).reshape(-1)
        if apply_gate and (p<self.presence_gate or q<self.quality_gate):
            return None,p,q
        return lm,p,q


class LandmarkerGestureRuntime:
    """Standalone landmarker -> frozen tiny_gesture crop+sequence towers."""
    STATUS_PREFIX="LAND"
    def __init__(self, conf=.6, presence_gate=.5, quality_gate=.15):
        import onnxruntime as ort
        self.lm=LandmarkerRuntime(presence_gate,quality_gate)
        self.crop=ort.InferenceSession(str(GESTURE/"crop.onnx"),providers=["CPUExecutionProvider"])
        self.seq=ort.InferenceSession(str(GESTURE/"seq.onnx"),providers=["CPUExecutionProvider"])
        self.meta=json.loads((GESTURE/"meta.json").read_text()); self.conf=conf
        self.presence_gate=presence_gate; self.quality_gate=quality_gate; self.ring=[]; self.last_lm=None

    def step(self,rgb):
        import cv2
        from train.gesture_lab import CLASSES6, CROP, SEQ_T, _lm_crop_box, _take_crop
        lm,presence,quality=self.lm.step(rgb)
        if lm is None:
            self.ring.clear(); self.last_lm=None; return "noop",1.,None
        self.last_lm=lm; cx,cy,side=_lm_crop_box(lm); crop=_take_crop(rgb,cx,cy,side)
        xc=(cv2.resize(crop,(CROP,CROP)).astype(np.float32)/255-.5).transpose(2,0,1)[None]
        probs=self.crop.run(["probabilities"],{"frames":xc})[0][0]
        self.ring.append(lm); self.ring=self.ring[-SEQ_T:]
        if len(self.ring)==SEQ_T:
            probs=(probs+self.seq.run(["probabilities"],{"frames":np.stack(self.ring)[None].astype(np.float32)})[0][0])/2
        top=int(probs.argmax()); return (CLASSES6[top],float(probs[top]),lm) if probs[top]>=self.conf else ("noop",float(probs[top]),lm)


def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=("train","eval","all"));
    p.add_argument("--source",action="append",dest="sources"); p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--batch",type=int,default=64); p.add_argument("--patience",type=int,default=8)
    p.add_argument("--device",choices=("auto","mps","cpu"),default="auto")
    p.add_argument("--data-fraction",type=float,default=1.)
    a=p.parse_args(); a.sources=a.sources or list(SOURCES)
    if a.stage in ("train","all"): train(a)
    if a.stage in ("eval","all"):
        imgs,lms,has,known,val,_=_load(a.sources,a.data_fraction); import cv2
        resized=np.stack([cv2.resize(x,(INPUT,INPUT)) for x in imgs[val]])
        print(json.dumps(evaluate_onnx(resized,lms[val],has[val],known[val]),indent=2))
if __name__=="__main__": main()
