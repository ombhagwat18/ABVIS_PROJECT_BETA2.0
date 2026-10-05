"""The one decision layer: per-bottle AI evidence -> PASS / REJECT / FAULT.

    Camera frames -> AI stage(s) (classification / detection / segmentation) -> Evidence
                  -> decide() -> Decision -> machine_cycle (FIFO) -> PLCService (M0 / M1)

No AI stage talks to the PLC and no stage decides on its own. Each stage turns ONE frame into
findings (defect names); decide() votes those findings across the frames captured for this
bottle and fuses cameras. Pure functions, no I/O: every rule is tested in demo().

Rules (all thresholds live in RULES and can be overridden from settings.json "decision_rules"):
  * Anything that stops us knowing the bottle is good is FAULT, never PASS: a camera fault, no
    frame after the trigger, an AI runtime error, a non-finite score, a required model missing,
    or no bottle found by detection/segmentation (cannot inspect what we cannot see).
  * Classification: a defect fires when its probability >= its project threshold (infer.decide).
  * Detection (components only, see detect.py), judged against the project's inspection recipe
    (default_recipe below = this bottle line): with a bottle found at the station, no cap box in
    the bottle's column -> missing_cap; cap centre below the top `cap_top_fraction` of the bottle
    -> cap_misplaced; no label box inside the bottle -> missing_label. A missing box only becomes a
    defect through the frame vote below, never from one frame.
  * Segmentation: label area / bottle area below `min_label_area_ratio` -> label_area_low; label
    mask filling less than `min_label_fill` of its own bounding box -> label_damaged; no cap mask
    on a segmented bottle -> missing_cap.
  * Vote per camera over the frames that saw a bottle: "majority" (default) needs a defect in at
    least half of them (ties reject), "any" in one, "all" in every one.
  * Cameras fuse FAULT > REJECT > PASS (same rule as infer.CameraSet.combined).
The detection/segmentation thresholds are development defaults, NOT validated on the machine.

    python decision.py     # self-test
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from infer import FAULT, PASS, REJECT, decide as classify_decide

CLASSIFICATION, DETECTION, SEGMENTATION = "classification", "detection", "segmentation"
# Task -> the AI stages it runs. The GUI shows the keys of TASK_LABELS.
TASKS = {
    "classification": (CLASSIFICATION,),
    "detection": (DETECTION,),
    "classification+detection": (CLASSIFICATION, DETECTION),
    "segmentation": (SEGMENTATION,),
}
TASK_LABELS = {"Classification": "classification", "Detection": "detection",
               "Classification + Detection": "classification+detection", "Segmentation": "segmentation"}

RULES = {
    "vote": "majority",              # any | majority | all
    "det_min_conf": 0.25,            # component boxes below this are ignored (DEVELOPMENT value)
    "station_x": 0.5,                # the inspected bottle is the one nearest this frame-width fraction
    "cap_top_fraction": 0.5,         # cap centre must sit in this top fraction of the bottle box
    "cap_above_margin": 0.25,        # ...or up to this fraction of bottle height ABOVE the box
    "min_label_area_ratio": 0.15,    # segmentation: label area / bottle area
    "min_label_fill": 0.80,          # segmentation: label area / label bounding-box area
    "check_margin": 0.0,             # >0: a classification score this close BELOW its threshold is "unsure"; a
                                     # bottle whose vote is only unsure scores is FAULT ("check"), never PASS
}


@dataclass
class FrameEvidence:
    """What the AI stages produced for one frame. *_error set = that stage failed on this frame."""
    camera: str
    frame_seq: int | None = None
    frame_ts: float | None = None
    probs: dict | None = None        # classification
    cls_error: str | None = None
    det: object = None               # detect.DetectionResult
    det_error: str | None = None
    seg: object = None               # segment.SegmentationResult
    seg_error: str | None = None


@dataclass
class CameraEvidence:
    camera: str
    frames: list = field(default_factory=list)
    fault: str | None = None         # camera-level problem: disconnected, no frame in the window...


@dataclass
class Decision:
    state: str                        # PASS | REJECT | FAULT
    defects: list = field(default_factory=list)
    reason: str = ""
    confidence: float | None = None
    per_camera: dict = field(default_factory=dict)   # camera -> (state, defects/reason, confidence)
    task: str = ""

    @property
    def defect_text(self) -> str:
        return ", ".join(self.defects) if self.defects else ("-" if self.state == PASS else self.reason)


# ----------------------------------------------------------------------------------- per-frame stages
def _centre(d):
    return (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2


# ------------------------------------------------------------------------ inspection recipe
# What "a complete product" means to the detection stage, per project (config.json "inspection"),
# so a new product needs a dataset, a trained detector and a recipe -- not a code change:
#
#   {"anchor": "bottle",                       the object inspected; none at the station -> FAULT
#    "parts": [{"name": "cap",                 a detector class that must belong to the anchor
#               "required": true,              missing -> defect "missing_<name>" (or "missing")
#               "search": [-0.25, 1.0],        where to look: centre y, as fractions of the anchor
#                                              height from its top (centre x must be inside it)
#               "zone": [-0.25, 0.5],          optional: found outside it -> "<name>_misplaced"
#               "misplaced": "cap_misplaced"}, ...]}
#
# With no recipe, default_recipe() is the bottle/cap/label rule this line was built with.
def default_recipe(rules) -> dict:
    up = float(rules["cap_above_margin"])
    return {"anchor": "bottle",
            "parts": [{"name": "cap", "search": [-up, 1.0], "zone": [-up, float(rules["cap_top_fraction"])],
                       "misplaced": "cap_misplaced"},
                      {"name": "label", "search": [0.0, 1.0]}]}


def recipe_of(rules) -> dict:
    return rules.get("recipe") or default_recipe(rules)


def recipe_classes(recipe) -> tuple:
    """The detector classes a recipe needs (the model must have all of them)."""
    return (recipe["anchor"],) + tuple(p["name"] for p in recipe.get("parts", ()))


def recipe_problems(recipe) -> list:
    """Why a recipe cannot be used ([] = usable). Checked before the line starts, so a typo cannot
    turn into every product being rejected for a missing part."""
    if not isinstance(recipe, dict) or not isinstance(recipe.get("anchor"), str) or not recipe["anchor"]:
        return ["recipe needs an 'anchor' class name"]
    out, seen = [], {recipe["anchor"]}
    for i, p in enumerate(recipe.get("parts") or []):
        n = p.get("name") if isinstance(p, dict) else None
        if not isinstance(n, str) or not n:
            out.append(f"part {i + 1}: needs a 'name'")
            continue
        if n in seen:
            out.append(f"part {n!r}: listed twice (or same as the anchor)")
        seen.add(n)
        for k in ("search", "zone"):
            v = p.get(k)
            if v is not None and not (isinstance(v, (list, tuple)) and len(v) == 2
                                      and all(isinstance(x, (int, float)) for x in v) and v[0] < v[1]):
                out.append(f"part {n!r}: {k} must be [top, bottom] with top < bottom")
    return out


def detection_findings(det, rules) -> tuple:
    """(findings, confidence) for one frame's DetectionResult, judged against the recipe. findings
    is None if no anchor (bottle) was found: the frame cannot say anything about this product."""
    conf = float(rules["det_min_conf"])
    recipe = recipe_of(rules)
    keep = [d for d in det.detections if d.confidence >= conf]
    anchors = [d for d in keep if d.class_name == recipe["anchor"]]
    if not anchors:
        return None, None
    sx = float(rules["station_x"]) * det.frame_wh[0]
    b = min(anchors, key=lambda d: (abs(_centre(d)[0] - sx), -d.confidence))
    bh = b.y2 - b.y1
    found, scores = [], [b.confidence]
    judge = rules.get("judge")                 # per-camera role: only these parts are judged by this camera
    for part in recipe.get("parts", ()):
        if judge is not None and part["name"] not in judge:
            continue
        top, bottom = part.get("search") or (0.0, 1.0)
        hits = [d for d in keep if d.class_name == part["name"] and b.x1 <= _centre(d)[0] <= b.x2
                and b.y1 + top * bh <= _centre(d)[1] <= b.y1 + bottom * bh]
        if not hits:
            if part.get("required", True):
                found.append(part.get("missing") or f"missing_{part['name']}")
            continue
        best = max(hits, key=lambda d: d.confidence)
        scores.append(best.confidence)
        zone = part.get("zone")
        if zone and not (b.y1 + zone[0] * bh <= _centre(best)[1] <= b.y1 + zone[1] * bh):
            found.append(part.get("misplaced") or f"{part['name']}_misplaced")
    return found, (b.confidence if found else min(scores))


def segmentation_findings(seg, rules) -> tuple:
    m = seg.measurements()
    if m["bottle_area_px"] is None:
        return None, None
    found = []
    if m["cap_present"] is False:
        found.append("missing_cap")
    if m["label_area_ratio"] is not None and m["label_area_ratio"] < float(rules["min_label_area_ratio"]):
        found.append("label_area_low")
    if m["label_fill"] is not None and m["label_fill"] < float(rules["min_label_fill"]):
        found.append("label_damaged")
    return found, seg.best("bottle").confidence


def _vote_needed(n: int, vote: str) -> int:
    return {"any": 1, "all": n}.get(vote, max(1, math.ceil(n / 2)))


# ------------------------------------------------------------------------------------- one camera
def decide_camera(task: str, cam: CameraEvidence, thresholds: dict, rules: dict) -> tuple:
    """(state, defects, reason, confidence) for one camera's frames of this bottle."""
    stages = TASKS[task]
    if cam.fault:
        return FAULT, [], cam.fault, None
    if not cam.frames:
        return FAULT, [], "no frame captured for this bottle", None
    per_frame, confs = [], []
    for f in cam.frames:
        found, fconf = set(), []
        for stage, err in ((CLASSIFICATION, f.cls_error), (DETECTION, f.det_error), (SEGMENTATION, f.seg_error)):
            if stage in stages and err:
                return FAULT, [], f"{stage} failed: {err}", None
        if CLASSIFICATION in stages:
            if f.probs is None:
                return FAULT, [], "classification produced no scores", None
            st, hits = classify_decide(f.probs, thresholds)
            if st == FAULT:
                return FAULT, [], "classification scores empty or not finite", None
            found.update(hits)
            m = float(rules.get("check_margin") or 0.0)
            if m > 0:
                for d, p in f.probs.items():
                    t = float(thresholds.get(d, 0.5))
                    if d not in hits and t - m <= p < t <= 1.0:
                        found.add("?" + d)               # unsure: close to the threshold
            hit_p = [f.probs[h] for h in hits]
            fconf.append(max(hit_p) if hits else 1.0 - max(f.probs.values()))
        saw_bottle = True
        if DETECTION in stages:
            if f.det is None:
                return FAULT, [], "detection produced no result", None
            dfound, dconf = detection_findings(f.det, rules)
            if dfound is None:
                saw_bottle = False
            else:
                found.update(dfound); fconf.append(dconf)
        if SEGMENTATION in stages:
            if f.seg is None:
                return FAULT, [], "segmentation produced no result", None
            sfound, sconf = segmentation_findings(f.seg, rules)
            if sfound is None:
                saw_bottle = False
            else:
                found.update(sfound); fconf.append(sconf)
        if saw_bottle:
            per_frame.append(found)
            confs.append(min(fconf) if fconf else None)
    if not per_frame:
        return FAULT, [], f"no bottle found in {len(cam.frames)} frame(s): cannot inspect", None
    need = _vote_needed(len(per_frame), rules.get("vote", "majority"))
    names = sorted({d for fr in per_frame for d in fr if not d.startswith("?")})
    defects = [d for d in names if sum(d in fr for fr in per_frame) >= need]
    if not defects:
        near = sorted({d.lstrip("?") for fr in per_frame for d in fr})
        unsure = [d for d in near if sum(d in fr or "?" + d in fr for fr in per_frame) >= need]
        if unsure:
            return FAULT, [], f"unsure: {', '.join(unsure)} scored just under the threshold (check the bottle)", None
    valid = [c for c in confs if c is not None]
    conf = sum(valid) / len(valid) if valid else None
    if defects:
        return REJECT, defects, f"{len(per_frame)} frame(s), vote {rules.get('vote', 'majority')}", conf
    return PASS, [], f"{len(per_frame)} frame(s)", conf


# -------------------------------------------------------------------------------------- the bottle
def decide(task: str, cameras: list, thresholds: dict | None = None, rules: dict | None = None) -> Decision:
    """All cameras' evidence for ONE bottle -> one Decision. No camera at all is FAULT."""
    r = dict(RULES)
    r.update(rules or {})
    if task not in TASKS:
        return Decision(FAULT, [], f"unknown inspection task {task!r}", None, {}, task)
    if not cameras:
        return Decision(FAULT, [], "no camera assigned to the inspection", None, {}, task)
    cam_rules = r.get("per_camera") or {}       # camera id -> overrides (station_x, judge, vote, ...)
    per = {c.camera: decide_camera(task, c, thresholds or {}, {**r, **(cam_rules.get(c.camera) or {})})
           for c in cameras}
    states = {v[0] for v in per.values()}
    state = FAULT if FAULT in states else REJECT if REJECT in states else PASS
    mine = {k: v for k, v in per.items() if v[0] == state}
    defects = sorted({d for v in mine.values() for d in v[1]})
    if state == FAULT:
        reason = "; ".join(f"{k}: {v[2]}" for k, v in mine.items())
    else:
        reason = "; ".join(f"{k}: {', '.join(v[1]) or 'ok'}" for k, v in per.items())
    confs = [v[3] for v in mine.values() if v[3] is not None]
    return Decision(state, defects, reason, min(confs) if confs else None,
                    {k: (v[0], v[1] or v[2], v[3]) for k, v in per.items()}, task)


# ------------------------------------------------------------------------------------------- tests
def demo():
    import numpy as np
    from detect import CLASS_NAMES as DET_CLASSES, Detection, DetectionResult
    from segment import FakeSeg, YoloSegmenter, rect

    def det(*boxes, wh=(400, 600), names=DET_CLASSES):
        out = tuple(Detection(list(names).index(n) if n in names else -1, n, c, *xyxy) for n, c, xyxy in boxes)
        return DetectionResult("0", 1, 0.0, out, wh, 5.0, "fake", 0.25)

    BOT = ("bottle", 0.95, (150, 100, 250, 550))
    CAP = ("cap", 0.90, (180, 80, 220, 130))
    LAB = ("label", 0.85, (155, 250, 245, 420))
    good, nocap, nolabel = det(BOT, CAP, LAB), det(BOT, LAB), det(BOT, CAP)
    lowcap = det(BOT, ("cap", 0.9, (180, 450, 220, 500)), LAB)                    # cap down at the bottle's base
    empty = det()

    def cam(*frames, name="cam0", **kw):
        return CameraEvidence(name, [FrameEvidence(name, i + 1, float(i), **f) for i, f in enumerate(frames)])

    D = lambda task, *cams, **kw: decide(task, list(cams), kw.get("thr", {}), kw.get("rules"))

    # detection
    assert D("detection", cam({"det": good}, {"det": good})).state == PASS
    d = D("detection", cam({"det": nocap}, {"det": nocap}, {"det": good}))
    assert d.state == REJECT and d.defects == ["missing_cap"], d
    d = D("detection", cam({"det": nolabel}))
    assert d.state == REJECT and d.defects == ["missing_label"] and abs(d.confidence - 0.95) < 1e-9, d
    assert D("detection", cam({"det": lowcap})).defects == ["cap_misplaced"]
    # one bad frame of three does not reject under majority, does under "any"
    assert D("detection", cam({"det": nocap}, {"det": good}, {"det": good})).state == PASS
    assert D("detection", cam({"det": nocap}, {"det": good}, {"det": good}), rules={"vote": "any"}).state == REJECT
    # 1 of 2 is a tie: rejects
    assert D("detection", cam({"det": nocap}, {"det": good})).state == REJECT
    # no bottle -> FAULT, never PASS and never a "missing" defect
    d = D("detection", cam({"det": empty}, {"det": empty}))
    assert d.state == FAULT and "no bottle" in d.reason and not d.defects, d
    # frames without a bottle are not votes: 1 empty + 1 missing-cap frame rejects on the frame that saw it
    assert D("detection", cam({"det": empty}, {"det": nocap})).defects == ["missing_cap"]
    # low-confidence boxes are ignored
    weak = det(BOT, ("cap", 0.1, CAP[2]), LAB)
    assert D("detection", cam({"det": weak})).defects == ["missing_cap"]
    # the bottle at the station (nearest station_x) is the one judged, not a neighbour
    two = det(("bottle", 0.99, (0, 100, 90, 550)), BOT, CAP, LAB)                 # capless neighbour at the edge
    assert D("detection", cam({"det": two})).state == PASS
    # the default recipe IS the bottle rule: spelling it out changes nothing
    explicit = {**RULES, "recipe": default_recipe(RULES)}
    for f in (good, nocap, nolabel, lowcap, weak, two, empty):
        assert detection_findings(f, explicit) == detection_findings(f, RULES)
    assert recipe_classes(default_recipe(RULES)) == ("bottle", "cap", "label")
    # another product, same code: a box that must carry a sticker in its top half, an optional tag
    box_recipe = {"anchor": "box", "parts": [{"name": "sticker", "search": [0.0, 1.0], "zone": [0.0, 0.5]},
                                             {"name": "tag", "required": False}]}
    assert recipe_problems(box_recipe) == []
    BOX = ("box", 0.9, (100, 100, 300, 500))

    def sticker(y):
        return ("sticker", 0.8, (150, y, 250, y + 40))

    def R(*b):
        return D("detection", cam({"det": det(*b, names=("box", "sticker", "tag"))}), rules={"recipe": box_recipe})
    assert R(BOX, sticker(150)).state == PASS                                  # optional tag absent: fine
    assert R(BOX).defects == ["missing_sticker"]
    assert R(BOX, sticker(400)).defects == ["sticker_misplaced"]
    assert R(sticker(150)).state == FAULT                                      # no anchor: cannot inspect
    assert R(BOT, CAP, LAB).state == FAULT                                     # a bottle is not this product
    assert recipe_problems({"parts": []})
    assert recipe_problems({"anchor": "box", "parts": [{"name": "box"}]})
    assert recipe_problems({"anchor": "box", "parts": [{"name": "s", "zone": [0.5, 0.1]}]})
    # runtime failures and missing results are FAULT
    assert D("detection", cam({"det": good}, {"det_error": "cuda oom"})).state == FAULT
    assert D("detection", cam({})).state == FAULT

    # classification
    thr = {"tilt_cap": 0.5, "missing_label": 0.5}
    ok_p, bad_p = {"tilt_cap": 0.1, "missing_label": 0.2}, {"tilt_cap": 0.9, "missing_label": 0.2}
    d = D("classification", cam({"probs": ok_p}, {"probs": ok_p}), thr=thr)
    assert d.state == PASS and abs(d.confidence - 0.8) < 1e-9, d
    d = D("classification", cam({"probs": bad_p}, {"probs": bad_p}), thr=thr)
    assert d.state == REJECT and d.defects == ["tilt_cap"] and abs(d.confidence - 0.9) < 1e-9, d
    assert D("classification", cam({"probs": {"tilt_cap": float("nan")}}), thr=thr).state == FAULT
    assert D("classification", cam({"probs": {}}), thr=thr).state == FAULT
    assert D("classification", cam({"cls_error": "boom"}), thr=thr).state == FAULT
    # check band: a score just under its threshold in most frames is FAULT (unsure), never PASS; off by default
    near_p = {"tilt_cap": 0.45, "missing_label": 0.1}
    assert D("classification", cam({"probs": near_p}, {"probs": near_p}), thr=thr).state == PASS
    d = D("classification", cam({"probs": near_p}, {"probs": near_p}), thr=thr, rules={"check_margin": 0.1})
    assert d.state == FAULT and "unsure: tilt_cap" in d.reason, d
    assert D("classification", cam({"probs": bad_p}, {"probs": bad_p}), thr=thr, rules={"check_margin": 0.1}).defects == ["tilt_cap"]
    # classification + detection: either stage can reject; a failure in either is FAULT
    d = D("classification+detection", cam({"probs": ok_p, "det": nocap}), thr=thr)
    assert d.state == REJECT and d.defects == ["missing_cap"], d
    d = D("classification+detection", cam({"probs": bad_p, "det": good}), thr=thr)
    assert d.defects == ["tilt_cap"], d
    assert D("classification+detection", cam({"probs": ok_p, "det_error": "x"}), thr=thr).state == FAULT

    # segmentation
    frame = np.zeros((600, 400, 3), np.uint8)
    items = {"v": [(0, 0.95, rect(150, 100, 250, 550)), (1, 0.9, rect(180, 80, 220, 130)),
                   (2, 0.85, rect(155, 250, 245, 420))]}
    seg = YoloSegmenter(model=FakeSeg(lambda f: items["v"]))
    assert D("segmentation", cam({"seg": seg.segment(frame)})).state == PASS
    items["v"] = [items["v"][0], items["v"][1], (2, 0.85, rect(155, 250, 245, 270))]       # label only a strip
    d = D("segmentation", cam({"seg": seg.segment(frame)}))
    assert d.state == REJECT and d.defects == ["label_area_low"], d
    items["v"] = [items["v"][0], items["v"][1], (2, 0.8, [(155, 250), (245, 250), (245, 420), (200, 300), (155, 420)])]
    assert "label_damaged" in D("segmentation", cam({"seg": seg.segment(frame)})).defects
    items["v"] = []
    assert D("segmentation", cam({"seg": seg.segment(frame)})).state == FAULT
    assert D("segmentation", cam({"seg_error": "no segmentation model trained"})).state == FAULT

    # camera faults and fusion
    assert D("detection", CameraEvidence("cam0", fault="camera disconnected")).state == FAULT
    assert D("detection", CameraEvidence("cam0")).state == FAULT                   # no frames
    assert decide("detection", []).state == FAULT
    assert decide("ocr", [cam({"det": good})]).state == FAULT
    c0, c1 = cam({"det": good}, name="cam0"), cam({"det": nolabel}, name="cam1")
    d = D("detection", c0, c1)
    assert d.state == REJECT and d.defects == ["missing_label"] and d.per_camera["cam0"][0] == PASS, d
    d = D("detection", c1, CameraEvidence("cam2", fault="frame timeout"))
    assert d.state == FAULT and "cam2: frame timeout" in d.reason, d                 # FAULT outranks REJECT
    # per-camera roles: cam1 is the cap/neck camera (judges only the cap), so the label it cannot see
    # from its side is not its finding; cam0 still judges everything
    roles = {"per_camera": {"cam1": {"judge": ["cap"]}}}
    assert D("detection", c0, c1, rules=roles).state == PASS
    c1cap = cam({"det": nocap}, name="cam1")
    d = D("detection", c0, c1cap, rules=roles)
    assert d.state == REJECT and d.defects == ["missing_cap"] and d.per_camera["cam1"][0] == REJECT, d
    # a per-camera station_x override picks that camera's bottle, not the neighbour
    two_r = det(("bottle", 0.99, (0, 100, 90, 550)), BOT, CAP, LAB)
    assert D("detection", cam({"det": two_r}, name="cam1"),
             rules={"per_camera": {"cam1": {"station_x": 0.05}}}).defects == ["missing_cap", "missing_label"]
    print("ok  decision engine: classification / detection (recipe-driven) / segmentation rules, frame vote, "
          "camera fusion FAULT > REJECT > PASS, every failure -> FAULT")


if __name__ == "__main__":
    demo()
