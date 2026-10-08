"""Checks a model must pass before it may become the production model (used by model_registry.validate).

shortcut_check(stamp)
    Has this classifier learned the BACKGROUND instead of the bottle? It is run on white-background bottles that DO
    have a cap and a label (Stage 2 scene images with a cap box). A model that calls those "missing_cap" (or any
    defect, for most of them) has learned "white background = defect" -- this happened on 2026-10-05 with the 22
    white-background missing-cap images. The images are only read; nothing is trained.

real_camera_gate(numbers, gates)
    The engineer's measured numbers on the REAL camera (bottles shown with TEST INSPECTION or on the line):
    good bottles called defective and defective bottles passed, each against a limit and a minimum sample size.

    python model_checks.py --selftest
    python model_checks.py shortcut <stamp>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE2 = ROOT / "stage2_dataset"

DEFAULT_GATES = {
    "max_good_called_defective_pct": 2.0,    # of real good bottles, called defective
    "max_defective_passed_pct": 1.0,         # of real defective bottles, passed as good
    "min_real_good": 30,                     # at least this many real good bottles shown
    "min_real_defective": 30,                # and this many real defective bottles
    "max_shortcut_fire_pct": 10.0,           # capped white-background bottles flagged missing_cap
}


def gates(settings: dict | None = None) -> dict:
    g = dict(DEFAULT_GATES)
    g.update((settings or {}).get("activation_gates") or {})
    return g


def capped_white_images(n: int = 25) -> list:
    """Stage 2 images (white-background session and others) whose ground truth has a bottle AND a cap."""
    ann = STAGE2 / "annotations_v2.json"
    if not ann.exists():
        return []
    a = json.loads(ann.read_text(encoding="utf-8"))["images"]
    out = []
    for f in sorted(a):
        cls = {b["cls"] for b in a[f]["boxes"]}
        if {"bottle", "cap"} <= cls and (STAGE2 / "clean" / f).exists():
            out.append(STAGE2 / "clean" / f)
    # prefer the white-background session (scene 22, img_3xx) where the shortcut was found
    white = [p for p in out if p.name.startswith("img_3")]
    return (white + [p for p in out if p not in white])[:n]


def shortcut_check(stamp: str, n: int = 25, predict=None, thresholds: dict | None = None) -> dict:
    """{'n', 'fired', 'fire_pct', 'any_defect_pct', 'passed'|'skipped'}. predict(bgr) -> {defect: prob} can be
    injected (self-test); by default the checkpoint `stamp` of the active project is loaded."""
    import cv2
    imgs = capped_white_images(n)
    if not imgs:
        return {"skipped": "no capped Stage 2 images on this PC (stage2_dataset/clean)", "n": 0}
    if predict is None:
        from vision import dataset as D
        from vision import infer
        m = infer.Model(stamp)
        predict = m.predict
        mp = D.MODELS / stamp / "metrics.json"
        if thresholds is None and mp.exists():
            thresholds = {d: v.get("threshold", 0.5) for d, v in json.loads(mp.read_text())["per_defect"].items()}
    thr = thresholds or {}
    fired = anyd = 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        probs = predict(img)
        if probs.get("missing_cap", 0.0) >= float(thr.get("missing_cap", 0.5)):
            fired += 1
        if any(v >= float(thr.get(d, 0.5)) for d, v in probs.items()):
            anyd += 1
    k = len(imgs)
    pct = 100.0 * fired / k
    return {"n": k, "fired": fired, "fire_pct": round(pct, 1), "any_defect_pct": round(100.0 * anyd / k, 1),
            "passed": pct <= gates(_settings())["max_shortcut_fire_pct"]}


def real_camera_gate(numbers: dict, g: dict | None = None) -> list:
    """Problems with the measured real-camera numbers ([] = they pass). numbers: good_n, good_called_defective,
    defective_n, defective_passed (counts)."""
    g = g or gates(_settings())
    try:
        gn, gd = int(numbers["good_n"]), int(numbers["good_called_defective"])
        dn, dp = int(numbers["defective_n"]), int(numbers["defective_passed"])
    except (KeyError, TypeError, ValueError):
        return ["real-camera numbers missing (good bottles shown / called defective, defective shown / passed)"]
    out = []
    if gn < g["min_real_good"]:
        out.append(f"only {gn} real good bottles shown (need >= {g['min_real_good']})")
    if dn < g["min_real_defective"]:
        out.append(f"only {dn} real defective bottles shown (need >= {g['min_real_defective']})")
    if gn and 100.0 * gd / gn > g["max_good_called_defective_pct"]:
        out.append(f"{gd}/{gn} good bottles called defective = {100.0 * gd / gn:.1f} % "
                   f"(limit {g['max_good_called_defective_pct']:g} %)")
    if dn and 100.0 * dp / dn > g["max_defective_passed_pct"]:
        out.append(f"{dp}/{dn} defective bottles passed = {100.0 * dp / dn:.1f} % (limit {g['max_defective_passed_pct']:g} %)")
    return out


def _settings() -> dict:
    try:
        from vision import dataset as D
        return D.load_settings()
    except Exception:                                    # noqa: BLE001
        return {}


def selftest():
    ok = {"good_n": 50, "good_called_defective": 1, "defective_n": 40, "defective_passed": 0}
    assert real_camera_gate(ok, DEFAULT_GATES) == []
    assert any("good bottles called defective" in x for x in real_camera_gate(dict(ok, good_called_defective=5), DEFAULT_GATES))
    assert any("defective bottles passed" in x for x in real_camera_gate(dict(ok, defective_passed=2), DEFAULT_GATES))
    assert any("real good bottles" in x for x in real_camera_gate(dict(ok, good_n=10), DEFAULT_GATES))
    assert real_camera_gate({}, DEFAULT_GATES)
    imgs = capped_white_images(5)
    if imgs:
        bad = shortcut_check("x", 5, predict=lambda img: {"missing_cap": 0.99, "tilt_cap": 0.1}, thresholds={"missing_cap": 0.5})
        good = shortcut_check("x", 5, predict=lambda img: {"missing_cap": 0.01, "tilt_cap": 0.1}, thresholds={"missing_cap": 0.5})
        assert not bad["passed"] and bad["fired"] == bad["n"] and good["passed"] and good["fired"] == 0, (bad, good)
        note = f"shortcut check on {len(imgs)} capped Stage 2 images"
    else:
        note = "shortcut check skipped (no Stage 2 images here)"
    print(f"ok  model_checks: real-camera gates (limits + minimum sample sizes), {note}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--selftest":
        selftest()
    elif a[0] == "shortcut":
        print(json.dumps(shortcut_check(a[1]), indent=2))
