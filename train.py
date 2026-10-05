"""Multi-label fine-tune. One sigmoid per defect, BCE with pos_weight.

Accuracy is not reported on purpose: a model that answers "no defect" to
everything scores 97% accuracy on missing_label. Recall is what matters here --
a missed defect ships.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import dataset as D


class BottleDS(Dataset):
    def __init__(self, paths, labels, defects, cfg, train: bool):
        self.paths, self.labels, self.defects, self.cfg = paths, labels, defects, cfg
        self.train = train
        self.tw, self.th = D.input_wh(cfg)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rel = self.paths[i]
        img = D.cached_crop(rel, self.cfg)
        if img is None:
            cw, ch = D.cache_wh(self.cfg)
            img = np.zeros((ch, cw, 3), np.uint8)
        img = self._aug(img) if self.train else self._center(img)
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / \
            torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        y = torch.tensor([float(self.labels[rel].get(d, 0)) for d in self.defects])
        return x, y

    def _center(self, img):
        return D.center_crop(img, (self.tw, self.th))

    def _aug(self, img):
        # No horizontal flip: it mirrors the label text and inverts skew
        # direction, which is exactly what skewed_label/skewed_bottle encode.
        h, w = img.shape[:2]
        ang = np.random.uniform(-5, 5)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, np.random.uniform(0.97, 1.03))
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        my = np.random.randint(0, max(1, h - self.th + 1))
        mx = np.random.randint(0, max(1, w - self.tw + 1))
        img = img[my:my + self.th, mx:mx + self.tw]
        img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2)
                      + np.random.uniform(-18, 18), 0, 255).astype(np.uint8)
        return img


DEFAULT_ARCH = "efficientnet_b0"

BACKBONES = {
    "efficientnet_b0": lambda tv, w: tv.models.efficientnet_b0(weights=w),
    "efficientnet_b1": lambda tv, w: tv.models.efficientnet_b1(weights=w),
    "mobilenet_v3_small": lambda tv, w: tv.models.mobilenet_v3_small(weights=w),
    "resnet18": lambda tv, w: tv.models.resnet18(weights=w),
    "convnext_tiny": lambda tv, w: tv.models.convnext_tiny(weights=w),
}

# Every backbone above ends in a final nn.Linear, but not at the same
# attribute -- ResNet keeps a bare .fc, the rest bury it inside .classifier
# at a family-specific index.
_CLASSIFIER_INDEX = {"efficientnet_b0": 1, "efficientnet_b1": 1,
                     "mobilenet_v3_small": 3, "convnext_tiny": 2}


def build(arch: str, n_out: int, pretrained: bool = True):
    """pretrained=False skips the ImageNet download -- used at inference load
    time, where the checkpoint's own state_dict overwrites every weight
    anyway, so fetching ImageNet weights first would be a wasted network call
    on every model switch."""
    import torchvision
    if arch not in BACKBONES:
        raise ValueError(f"unknown backbone {arch!r} -- choices: {', '.join(BACKBONES)}")
    weights = "IMAGENET1K_V1" if pretrained else None
    m = BACKBONES[arch](torchvision, weights)
    if arch == "resnet18":
        m.fc = nn.Linear(m.fc.in_features, n_out)
    else:
        i = _CLASSIFIER_INDEX[arch]
        m.classifier[i] = nn.Linear(m.classifier[i].in_features, n_out)
    return m


def _bench_speed(model, dev, input_wh, n=30):
    """Average single-image inference latency (ms) and FPS, timed on the
    device the model actually trained on."""
    w, h = input_wh
    x = torch.zeros(1, 3, h, w, device=dev)
    model.eval()
    with torch.no_grad():
        for _ in range(5):          # warm up: first calls pay setup cost
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
    ms = (time.time() - t0) / n * 1000
    return round(ms, 2), round(1000 / ms, 1) if ms else 0.0


def _pr(prob: np.ndarray, truth: np.ndarray, thr: float):
    pred = prob >= thr
    tp = int((pred & (truth == 1)).sum())
    fp = int((pred & (truth == 0)).sum())
    fn = int((~pred & (truth == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn


def evaluate(probs, truths, defects, n_train_pos=None):
    """Per-defect precision/recall, plus the threshold that maximises F1.

    A defect with no validation positives gets no score at all rather than a
    0.000 that reads like a failure. "No images anywhere" and "images exist but
    none reached validation" are different problems with different fixes.
    """
    out = {}
    for i, d in enumerate(defects):
        pr, tr = probs[:, i], truths[:, i]
        n_pos = int(tr.sum())
        if n_pos == 0:
            tp = (n_train_pos or {}).get(d, 0)
            # A defect with no training images is disabled outright (threshold
            # above 1.0, unreachable). Its logit was never constrained by a
            # single positive example, so on anything unfamiliar it will happily
            # read 0.8 -- and hand an operator a reject reason that has never
            # been seen. No data means no claim.
            out[d] = {"n_pos": 0, "n_train_pos": tp, "threshold": 0.5 if tp else 1.01,
                      "precision": 0.0, "recall": 0.0, "f1": 0.0,
                      "note": ("no images of this defect at all - disabled until some exist"
                               if tp == 0 else
                               f"{tp} training images but none in validation - unscored")}
            continue
        best = max(((_pr(pr, tr, t), t) for t in np.arange(0.05, 0.96, 0.05)),
                   key=lambda z: z[0][2])
        (p, r, f, tp, fp, fn), thr = best
        out[d] = {"n_pos": n_pos, "threshold": round(float(thr), 2),
                  "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                  "tp": tp, "fp": fp, "fn": fn}
    return out


def _score_against(stamp: str, stored_key: str, noun: str, log=print, should_stop=None) -> dict:
    """Shared machinery behind score_checkpoint() and evaluate_test(): load a
    checkpoint's own recorded path list (val_paths or test_paths), score it
    against CURRENT labels, and refuse rather than silently substitute today's
    split if that list was never recorded.

    The checkpoint's own ROI and input size are used, not config.json's, so a
    model is always fed the crop it was trained on. Re-deriving the split from
    today's labels/folders instead of the recorded path list would hand a
    model a share of its own training images and report a perfect 1.000 for
    every defect -- the split moves whenever a class is renamed or an image
    reviewed, so only the exact list recorded at training time is honest.
    """
    import infer

    mp = D.MODELS / stamp / "metrics.json"
    stored = json.loads(mp.read_text()) if mp.exists() else {}
    wanted = stored.get(stored_key)
    if not wanted:
        raise RuntimeError(
            f"this checkpoint did not record its {noun} set, so it cannot be "
            f"scored without reusing images it trained on - retrain to include it")

    m = infer.Model(stamp)
    defects, all_labels = D.load_labels()
    labels = {p: r for p, r in all_labels.items() if r.get("reviewed", 1)}
    if not labels:
        raise RuntimeError("no reviewed images to score against")
    use_paths = [p for p in wanted if p in labels]
    if not use_paths:
        raise RuntimeError(f"none of this model's {noun} images still exist")

    # Only defects the checkpoint actually has an output for can be scored. A
    # class added after this model trained is not a failure of the model.
    shared = [d for d in m.defects if d in defects]
    missing = [d for d in defects if d not in m.defects]

    P, T, used = [], [], []
    for i, p in enumerate(use_paths):
        if should_stop is not None and should_stop():
            break
        img = D.cached_crop(p, m.cfg)
        if img is None:
            continue
        probs = m.predict_view(D.center_crop(img, m.input_wh))
        P.append([probs[d] for d in shared])
        T.append([float(labels[p].get(d, 0)) for d in shared])
        used.append(p)
        if i % 25 == 0 and i:
            log(f"  {i}/{len(use_paths)}")

    if not P:
        raise RuntimeError(f"no {noun} images could be read")
    probs, truths = np.array(P, np.float32), np.array(T, np.float32)
    held = set(used)
    train_pos = {d: sum(labels[p].get(d, 0) for p in labels if p not in held) for d in shared}
    met = evaluate(probs, truths, shared, train_pos)
    scored = [x for x in met.values() if x["n_pos"]]
    return {"stamp": stamp, "n_scored": len(used), "defects": shared,
            "missing_defects": missing, "per_defect": met,
            "dropped": len(wanted) - len(used),
            "macro_f1": round(float(np.mean([x["f1"] for x in scored])), 4) if scored else 0.0}


def score_checkpoint(stamp: str, log=print, should_stop=None) -> dict:
    """Re-score a saved checkpoint's VALIDATION set against the CURRENT labels.

    A model's stored metrics were measured against the labels as they stood the
    day it trained. Once anyone relabels, two checkpoints' stored numbers are no
    longer comparable -- they were marked by different examiners. Comparing
    models means running them all over one split, today. Slow (a forward pass
    per validation image) and worth it: the alternative is a league table whose
    rows do not share a ruler.
    """
    r = _score_against(stamp, "val_paths", "validation", log, should_stop)
    return {**r, "n_val": r.pop("n_scored")}


def evaluate_test(stamp: str, log=print, should_stop=None) -> dict:
    """Score a checkpoint against its genuinely HELD-OUT test set.

    Unlike score_checkpoint(), this is not meant to be run after every
    training pass or used to pick a "best" model -- test_paths was carved out
    before training/validation ever saw it (dataset.split_train_val_test) and
    scoring it repeatedly to chase a better number would make it just another
    validation set. Call this once, deliberately, for a final reported number.
    """
    r = _score_against(stamp, "test_paths", "held-out test", log, should_stop)
    return {**r, "n_test": r.pop("n_scored")}


def run(epochs=25, batch=32, lr=3e-4, arch=DEFAULT_ARCH, test_frac=0.15,
        log=print, on_epoch=None, patience=None, activate=None) -> dict:
    """Train one checkpoint. It is registered as a CANDIDATE and the production model is NOT replaced
    unless activate=True (or settings.json "auto_activate_trained_model": true). Activation normally
    goes through model_registry (validate -> approve -> activate), which keeps a rollback record."""
    if activate is None:
        activate = bool(D.load_settings().get("auto_activate_trained_model", False))
    cfg = D.load_config()
    defects, all_labels = D.load_labels()
    if not all_labels:
        raise RuntimeError(f"no images in project {D.PROJECT!r} -- upload some first")

    # unreviewed images are all-zero by default -- training on them would teach
    # the model that every unlabelled capture is a good bottle
    labels = {p: r for p, r in all_labels.items() if r.get("reviewed", 1)}
    skipped = len(all_labels) - len(labels)
    if skipped:
        log(f"skipping {skipped} unreviewed image(s) - label them in the inbox first")
    if not labels:
        raise RuntimeError("no reviewed images to train on")

    log("grouping near-duplicate frames into scenes...")
    # test_paths is a genuinely held-out set: carved out before train/val ever
    # sees it, never used for training or for picking the best epoch/checkpoint.
    # It exists so a final, honest number can be produced once, later -- not
    # every run -- via evaluate_test(). See dataset.split_train_val_test.
    tr_paths, va_paths, test_paths = D.split_train_val_test(sorted(labels), test_frac=test_frac)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_scenes = len(set(D.scene_map(sorted(labels)).values()))
    log(f"device={dev}  defects={len(defects)}  train={len(tr_paths)}  val={len(va_paths)}  "
        f"test(held out)={len(test_paths)}")
    log(f"{len(labels)} images are ~{n_scenes} distinct scenes - that is the real "
        f"sample size, and the ceiling on how much these scores can mean")

    cnt = D.counts(defects, labels)
    for d in defects:
        if cnt[d] == 0:
            log(f"  ! {d}: 0 positive images - it cannot be learned, add some")
    log(f"  good (no defect flags): {cnt['_good']}")

    log("caching crops (first run only, this is the slow part)...")
    for i, p in enumerate(tr_paths + va_paths):
        D.cached_crop(p, cfg)
        if i % 200 == 0 and i:
            log(f"  cached {i}/{len(labels)}")

    tr = DataLoader(BottleDS(tr_paths, labels, defects, cfg, True), batch_size=batch,
                    shuffle=True, num_workers=0, drop_last=len(tr_paths) > batch)
    va = DataLoader(BottleDS(va_paths, labels, defects, cfg, False), batch_size=batch)

    model = build(arch, len(defects)).to(dev)
    train_pos = {d: sum(labels[p].get(d, 0) for p in tr_paths) for d in defects}
    pos = np.array([train_pos[d] for d in defects], np.float32)
    neg = len(tr_paths) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 50), dtype=torch.float32, device=dev)
    log("pos_weight: " + ", ".join(f"{d}={w:.1f}" for d, w in zip(defects, pw.tolist())))

    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")

    best_f1, best_state, best_metrics, best_probs, best_epoch = -1.0, None, None, None, 0
    no_improve = 0                # consecutive epochs without a validation macro-F1 gain
    history = []                  # per-epoch, so the curves can be drawn later
    for ep in range(1, epochs + 1):
        model.train()
        tot, t0 = 0.0, time.time()
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev, enabled=dev == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
        sched.step()

        model.eval()
        P, T = [], []
        with torch.no_grad():
            for x, y in va:
                with torch.amp.autocast(dev, enabled=dev == "cuda"):
                    P.append(torch.sigmoid(model(x.to(dev))).float().cpu().numpy())
                T.append(y.numpy())
        probs, truths = np.concatenate(P), np.concatenate(T)
        met = evaluate(probs, truths, defects, train_pos)
        trainable = [m for m in met.values() if m["n_pos"]]
        mf1 = float(np.mean([m["f1"] for m in trainable])) if trainable else 0.0
        ep_loss = tot / max(1, len(tr_paths))
        log(f"epoch {ep:>3}/{epochs}  loss={ep_loss:.4f}  "
            f"macro-F1={mf1:.3f}  ({time.time() - t0:.1f}s)")
        history.append({
            "epoch": ep, "loss": round(ep_loss, 5), "macro_f1": round(mf1, 4),
            "seconds": round(time.time() - t0, 2),
            "lr": round(float(opt.param_groups[0]["lr"]), 8),
            # per-defect too: a macro average hides one class collapsing while
            # the rest improve, which is the failure this dataset is prone to
            "recall": {d: met[d]["recall"] for d in defects},
            "precision": {d: met[d]["precision"] for d in defects}})
        if on_epoch:
            on_epoch(history[-1])
        if mf1 > best_f1:
            best_f1 = mf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics, best_probs, best_epoch = met, probs, ep
            no_improve = 0
        else:
            no_improve += 1

        # Opt-in only (patience=None keeps every existing call, including the
        # CLI with no --patience, running the full fixed epoch count exactly
        # as before). Monitors validation macro-F1 only -- the held-out test
        # set is never read here, so it cannot influence when training stops
        # or which epoch's weights get kept.
        if patience is not None and no_improve >= patience:
            log(f"no validation macro-F1 improvement for {patience} epochs -- "
                f"stopping early at epoch {ep} (best was epoch {best_epoch})")
            break

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = D.MODELS / stamp
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "defects": defects, "input_wh": list(D.input_wh(cfg)),
                "cache_wh": list(D.cache_wh(cfg)), "project": D.PROJECT,
                "roi": cfg.get("roi"), "roi_frame": cfg.get("roi_frame"),
                "state_dict": best_state}, out / "model.pt")
    model_size_mb = round((out / "model.pt").stat().st_size / (1024 * 1024), 2)
    infer_ms, fps = _bench_speed(model, dev, D.input_wh(cfg))

    # Informational aggregates for cross-architecture benchmarking. macro_f1
    # (above) stays the primary number this file's docstring already argues
    # for -- plain accuracy is misleading under the class imbalance this
    # dataset has, so it is reported but not treated as the headline metric.
    scored_vals = [x for x in best_metrics.values() if x["n_pos"]]
    macro_precision = round(float(np.mean([x["precision"] for x in scored_vals])), 4) if scored_vals else 0.0
    macro_recall = round(float(np.mean([x["recall"] for x in scored_vals])), 4) if scored_vals else 0.0
    accuracy = round(float(np.mean([
        (x["tp"] + max(0, len(va_paths) - x["tp"] - x["fp"] - x["fn"])) / max(1, len(va_paths))
        for x in scored_vals])), 4) if scored_vals else 0.0

    # The labels this model was trained on, beside the model. Rolling back to an
    # older checkpoint without them rolls back the weights and keeps whatever
    # labelling has happened since, which is not the state that scored these
    # metrics -- and relabelling is exactly what happens between runs.
    if D.LABELS_CSV.exists():
        shutil.copy2(D.LABELS_CSV, out / "labels.csv")

    # per-image val predictions -> the "show me the mistakes" screen
    mistakes = []
    for j, p in enumerate(va_paths):
        for i, d in enumerate(defects):
            t = int(truths[j, i]) if best_probs is not None else 0
            pr = float(best_probs[j, i])
            if (pr >= best_metrics[d]["threshold"]) != bool(t):
                mistakes.append({"path": p, "defect": d, "prob": round(pr, 3),
                                 "truth": t, "kind": "false_neg" if t else "false_pos"})
    mistakes.sort(key=lambda m: -abs(m["prob"] - 0.5))

    summary = {"stamp": stamp, "macro_f1": round(best_f1, 4), "defects": defects,
               "n_train": len(tr_paths), "n_val": len(va_paths), "epochs": epochs,
               "arch": arch, "batch": batch, "lr": lr, "project": D.PROJECT,
               "input_wh": list(D.input_wh(cfg)), "device": dev,
               "n_scenes": n_scenes, "history": history,
               "macro_precision": macro_precision, "macro_recall": macro_recall,
               "accuracy": accuracy, "infer_ms": infer_ms, "fps": fps,
               # additive, informational only -- absent/None when patience
               # isn't used, so nothing that reads older checkpoints breaks
               "patience": patience, "best_epoch": best_epoch,
               "epochs_run": len(history),
               "stopped_early": patience is not None and len(history) < epochs,
               "model_size_mb": model_size_mb,
               # The exact validation set, so this model can be re-scored later
               # without being handed its own training images. The split is
               # derived from the path set and the folder layout, both of which
               # move -- renaming a class or migrating the tree silently changes
               # which images land in validation, and a model scored on images
               # it memorised reports a perfect 1.000 that means nothing.
               "val_paths": va_paths,
               # Held out before training ever started, never touched for
               # training or for picking the best epoch/threshold. Scored only
               # by evaluate_test(), on request, not as part of this run.
               "n_test": len(test_paths), "test_paths": test_paths,
               "per_defect": best_metrics, "mistakes": mistakes[:300]}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    if activate:
        cfg["thresholds"] = {d: best_metrics[d]["threshold"] for d in defects}
        cfg["active_model"] = stamp
        D.save_config(cfg)
        log(f"activated {stamp} (auto-activate on): it is now the production classifier")
    else:
        log(f"{stamp} saved as a CANDIDATE: the production model was NOT changed "
            f"(validate / approve / activate it under Models)")

    log("")
    log(f"{'defect':<18}{'val+':>5}{'thr':>6}{'prec':>7}{'recall':>8}{'F1':>7}   note")
    for d in defects:
        m = best_metrics[d]
        if m["n_pos"]:
            log(f"{d:<18}{m['n_pos']:>5}{m['threshold']:>6}{m['precision']:>7.3f}"
                f"{m['recall']:>8.3f}{m['f1']:>7.3f}")
        else:
            log(f"{d:<18}{0:>5}{'':>6}{'-':>7}{'-':>8}{'-':>7}   {m['note']}")
    n_scored = sum(1 for m in best_metrics.values() if m["n_pos"])
    log(f"\nsaved models/{stamp}  macro-F1={best_f1:.3f} over "
        f"{n_scored}/{len(defects)} scored defect(s)  ({len(mistakes)} val mistakes)")
    return summary


def demo():
    """Self-check on the metric maths -- the part that is easy to get silently wrong."""
    truth = np.array([[1], [1], [0], [0]], np.float32)
    prob = np.array([[0.9], [0.4], [0.6], [0.1]], np.float32)
    p, r, f, tp, fp, fn = _pr(prob[:, 0], truth[:, 0], 0.5)
    assert (tp, fp, fn) == (1, 1, 1), (tp, fp, fn)
    assert abs(p - 0.5) < 1e-9 and abs(r - 0.5) < 1e-9

    m = evaluate(prob, truth, ["x"])["x"]
    assert m["threshold"] <= 0.4 and m["recall"] == 1.0, m  # sweep finds the perfect split

    # a column with no validation positives must not report a fake score, and
    # must say WHICH of the two problems it is
    z, o = np.array([[0.9]], np.float32), np.array([[0.0]], np.float32)
    none_at_all = evaluate(z, o, ["e"], {"e": 0})["e"]
    unscored = evaluate(z, o, ["e"], {"e": 42})["e"]
    assert none_at_all["n_pos"] == 0 and "at all" in none_at_all["note"], none_at_all
    assert "42 training images" in unscored["note"], unscored
    assert none_at_all["note"] != unscored["note"], "the two cases must read differently"

    # a defect with no training images must be unable to fire, at any confidence
    import infer
    assert none_at_all["threshold"] > 1.0, none_at_all
    ok, hits = infer.verdict({"e": 0.99}, {"e": none_at_all["threshold"]})
    assert ok and not hits, "a defect with zero training images fired"
    assert unscored["threshold"] <= 1.0, "an unscored-but-trained defect stays live"

    # Re-scoring must refuse a checkpoint that never recorded its validation
    # set. Scoring it on today's split hands it images it trained on and comes
    # back 1.000 on every defect -- the exact fake score the scene split exists
    # to prevent, arriving through the back door of the comparison table.
    import json as _json
    import tempfile
    was = D.MODELS
    try:
        with tempfile.TemporaryDirectory() as td:
            D.MODELS = Path(td)
            (D.MODELS / "old").mkdir()
            (D.MODELS / "old" / "metrics.json").write_text(_json.dumps({"macro_f1": 0.97}))
            try:
                score_checkpoint("old", log=lambda _m: None)
            except RuntimeError as e:
                assert "validation set" in str(e), e
            else:
                raise AssertionError("scored a checkpoint that recorded no validation set")
            # same guard, same reason, for the held-out test set
            try:
                evaluate_test("old", log=lambda _m: None)
            except RuntimeError as e:
                assert "held-out test set" in str(e), e
            else:
                raise AssertionError("scored a checkpoint that recorded no test set")
    finally:
        D.MODELS = was
    print("ok")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        n = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 25
        a = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else DEFAULT_ARCH
        pat = int(sys.argv[sys.argv.index("--patience") + 1]) if "--patience" in sys.argv else None
        run(epochs=n, arch=a, patience=pat, activate=True if "--activate" in sys.argv else None)
