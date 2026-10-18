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

from vision.infer import FAULT, PASS, REJECT, decide as classify_decide

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
    "det_min_conf_by_class": {"bottle": 0.55, "cap": 0.40, "label": 0.40},   # stricter per class; others use det_min_conf
    "bottle_min_aspect": 1.3,        # an upright bottle box is at least this much taller than wide (0 = off)
    "bottle_min_area": 0.02,         # ...and covers at least this fraction of the frame (0 = off)
    "station_x": 0.5,                # the inspected bottle is the one nearest this frame-width fraction
    "cap_top_fraction": 0.5,         # cap centre must sit in this top fraction of the bottle box
    "cap_above_margin": 0.25,        # ...or up to this fraction of bottle height ABOVE the box
    "min_label_area_ratio": 0.15,    # segmentation: label area / bottle area
    "min_label_fill": 0.80,          # segmentation: label area / label bounding-box area
    "check_margin": 0.0,             # >0: a classification score this close BELOW its threshold is "unsure"; a
                                     # bottle whose vote is only unsure scores is FAULT ("check"), never PASS
    # Three-zone decision (industrial confidence rule). A defect score >= its threshold is a DEFECT; <= pass_max is
    # confidently absent; anything in between is UNSURE, and a bottle the vote leaves unsure is FAULT (rejected),
    # never PASS. 0 = off. A PASS therefore needs every defect score low, not merely below its threshold.
    "pass_max": 0.0,
    # A camera must see the bottle (the recipe's anchor) in at least this share of its frames, else FAULT: a vote
    # over 1 bottle frame out of 13 is not an inspection. 0 = off (any one frame is enough).
    "min_bottle_frames_frac": 0.0,
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
    # WHERE each reported defect is: [(camera, frame_seq, finding, confidence, (x1, y1, x2, y2))], the most confident
    # box per camera and finding, whole-frame pixels. Only detector findings can be located (a classifier score is
    # whole-image); empty for PASS / FAULT.
    locations: list = field(default_factory=list)

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
    """The detector classes a recipe needs (the model must have all of them). Defect classes count too: a recipe that
    expects a cap_tilted box from a model that cannot produce one would silently never find a tilted cap."""
    out = [recipe["anchor"]]
    for p in recipe.get("parts", ()):
        out += [p["name"], *p.get("any_of", ())]
    out += [c for c in (recipe.get("defects") or {})]
    return tuple(dict.fromkeys(out))                     # ordered, no repeats


def recipe_defects(recipe) -> dict:
    """{detector class: (finding name, min confidence or None)} for the recipe's defect classes.
    "defects": {"cap_tilted": {"finding": "tilt_cap", "min_conf": 0.6}, "label_damaged": {}} -- a box of that class
    inside the inspected bottle IS the defect, located where the box is."""
    out = {}
    for cls, spec in (recipe.get("defects") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        out[cls] = (str(spec.get("finding") or cls), spec.get("min_conf"))
    return out


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
        alt = p.get("any_of", ())
        if not (isinstance(alt, (list, tuple)) and all(isinstance(a, str) and a for a in alt)):
            out.append(f"part {n!r}: any_of must be a list of class names")
        for k in ("search", "zone"):
            v = p.get(k)
            if v is not None and not (isinstance(v, (list, tuple)) and len(v) == 2
                                      and all(isinstance(x, (int, float)) for x in v) and v[0] < v[1]):
                out.append(f"part {n!r}: {k} must be [top, bottom] with top < bottom")
    defects = recipe.get("defects") or {}
    if not isinstance(defects, dict):
        return out + ["'defects' must be {class: {finding, min_conf}}"]
    for cls, spec in defects.items():
        if not isinstance(cls, str) or not cls or cls in seen:
            out.append(f"defect class {cls!r}: empty, or the same as the anchor / a part")
            continue
        if spec is not None and not isinstance(spec, dict):
            out.append(f"defect class {cls!r}: settings must be a dict")
            continue
        mc = (spec or {}).get("min_conf")
        if mc is not None and not (isinstance(mc, (int, float)) and 0.0 < mc <= 1.0):
            out.append(f"defect class {cls!r}: min_conf must be in (0, 1]")
    return out


def gate_detections(det, rules) -> tuple:
    """The boxes that count as a bottle on the line, nothing else. One bottle box (confident, bottle-shaped, the
    one nearest station_x) and, inside it, the parts that clear their own confidence. A clutter box (a laptop, a
    hand, a shelf) fails the shape or confidence test, so with no real bottle in view this returns () and the
    caller sees "no bottle", not a field of boxes. Used by the verdict AND by what the screen draws."""
    recipe = recipe_of(rules)
    by = rules.get("det_min_conf_by_class") or {}
    base = float(rules["det_min_conf"])
    W, H = det.frame_wh
    asp, area = float(rules.get("bottle_min_aspect") or 0), float(rules.get("bottle_min_area") or 0)

    def conf_ok(d):
        return d.confidence >= float(by.get(d.class_name, base))

    def shaped(d):
        w, h = max(1e-6, d.x2 - d.x1), max(1e-6, d.y2 - d.y1)
        return h / w >= asp and (w * h) / max(1.0, W * H) >= area
    anchors = [d for d in det.detections if d.class_name == recipe["anchor"] and conf_ok(d) and shaped(d)]
    if not anchors:
        return ()
    sx = float(rules["station_x"]) * W
    b = min(anchors, key=lambda d: (abs(_centre(d)[0] - sx), -d.confidence))
    bh = b.y2 - b.y1
    up = float(rules["cap_above_margin"]) * bh
    keep = [b]
    for d in det.detections:
        cx, cy = _centre(d)
        if d is not b and d.class_name != recipe["anchor"] and conf_ok(d) and b.x1 <= cx <= b.x2 and b.y1 - up <= cy <= b.y2:
            keep.append(d)
    return tuple(keep)


def detection_findings(det, rules) -> tuple:
    """(findings, confidence) for one frame's DetectionResult, judged against the recipe. findings
    is None if no anchor (bottle) was found: the frame cannot say anything about this product."""
    found, conf, _ = _judge(det, rules)
    return found, conf


def defect_locations(det, rules) -> list:
    """[(finding, confidence, (x1, y1, x2, y2))] in whole-frame pixels: WHERE each finding of this frame is.
    A defect-class box is located at the box; a missing part at the zone where it should have been; a misplaced
    part at the box that was found out of place. [] when no bottle was found."""
    return _judge(det, rules)[2]


def _judge(det, rules) -> tuple:
    """(findings or None, confidence, locations): the recipe applied to one frame."""
    recipe = recipe_of(rules)
    keep = list(gate_detections(det, rules))
    anchors = [d for d in keep if d.class_name == recipe["anchor"]]
    if not anchors:
        return None, None, []
    sx = float(rules["station_x"]) * det.frame_wh[0]
    b = min(anchors, key=lambda d: (abs(_centre(d)[0] - sx), -d.confidence))
    bh = b.y2 - b.y1
    found, scores, where = [], [b.confidence], []
    judge = rules.get("judge")                 # per-camera role: only these parts are judged by this camera
    for part in recipe.get("parts", ()):
        if judge is not None and part["name"] not in judge:
            continue
        top, bottom = part.get("search") or (0.0, 1.0)
        # any_of: classes that also ARE this part in a defective state (a cap_tilted box is still the cap: present,
        # so not "missing_cap"; the tilt itself is reported by the recipe's "defects" section)
        names = (part["name"], *part.get("any_of", ()))
        hits = [d for d in keep if d.class_name in names and b.x1 <= _centre(d)[0] <= b.x2
                and b.y1 + top * bh <= _centre(d)[1] <= b.y1 + bottom * bh]
        zone = part.get("zone")
        if not hits:
            if part.get("required", True):
                name = part.get("missing") or f"missing_{part['name']}"
                found.append(name)
                z = zone or (top, bottom)            # where the part should have been
                where.append((name, b.confidence, (b.x1, b.y1 + z[0] * bh, b.x2, b.y1 + z[1] * bh)))
            continue
        best = max(hits, key=lambda d: d.confidence)
        scores.append(best.confidence)
        if zone and not (b.y1 + zone[0] * bh <= _centre(best)[1] <= b.y1 + zone[1] * bh):
            name = part.get("misplaced") or f"{part['name']}_misplaced"
            found.append(name)
            where.append((name, best.confidence, (best.x1, best.y1, best.x2, best.y2)))
    base = float(rules["det_min_conf"])
    for cls, (name, min_conf) in recipe_defects(recipe).items():
        if judge is not None and cls not in judge and name not in judge:
            continue
        need = float(min_conf) if min_conf is not None else base
        hits = [d for d in keep if d.class_name == cls and d.confidence >= need]
        if hits:
            best = max(hits, key=lambda d: d.confidence)
            if name not in found:
                found.append(name)
            scores.append(best.confidence)
            where.append((name, best.confidence, (best.x1, best.y1, best.x2, best.y2)))
    W, H = det.frame_wh                        # a zone above the bottle can reach past the frame edge: clamp
    where = [(n, c, (min(max(x1, 0.0), W), min(max(y1, 0.0), H), min(max(x2, 0.0), W), min(max(y2, 0.0), H)))
             for n, c, (x1, y1, x2, y2) in where]
    return found, (b.confidence if found else min(scores)), where


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
            pmax = float(rules.get("pass_max") or 0.0)
            if m > 0 or pmax > 0:
                for d, p in f.probs.items():
                    t = float(thresholds.get(d, 0.5))
                    if d in hits or t > 1.0:             # a hit, or a disabled defect (threshold 1.01)
                        continue
                    if (m > 0 and t - m <= p < t) or (pmax > 0 and pmax < p < t):
                        found.add("?" + d)               # unsure: not low enough to call this defect absent
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
    frac = float(rules.get("min_bottle_frames_frac") or 0.0)
    if frac > 0 and len(per_frame) < frac * len(cam.frames):
        return (FAULT, [], f"bottle seen in only {len(per_frame)} of {len(cam.frames)} frame(s) "
                f"(need {frac:.0%}): cannot inspect", None)
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
    locations = []
    if state == REJECT and DETECTION in TASKS[task]:
        best = {}
        for c in cameras:
            if per[c.camera][0] != REJECT:
                continue
            crules = {**r, **(cam_rules.get(c.camera) or {})}
            for f in c.frames:
                if f.det is None:
                    continue
                for name, conf, box in defect_locations(f.det, crules):
                    k = (c.camera, name)
                    if name in per[c.camera][1] and (k not in best or conf > best[k][3]):
                        best[k] = (c.camera, f.frame_seq, name, float(conf), tuple(float(v) for v in box))
        locations = sorted(best.values(), key=lambda x: (x[0], x[2]))
    return Decision(state, defects, reason, min(confs) if confs else None,
                    {k: (v[0], v[1] or v[2], v[3]) for k, v in per.items()}, task, locations)


# ------------------------------------------------------------------------------------------- tests
def demo():
    import numpy as np
    from vision.detect import CLASS_NAMES as DET_CLASSES, Detection, DetectionResult
    from vision.segment import FakeSeg, YoloSegmenter, rect

    def det(*boxes, wh=(400, 600), names=DET_CLASSES):
        out = tuple(Detection(list(names).index(n) if n in names else -1, n, c, *xyxy) for n, c, xyxy in boxes)
        return DetectionResult("0", 1, 0.0, out, wh, 5.0, "fake", 0.25)

    BOT = ("bottle", 0.95, (150, 100, 250, 550))
    CAP = ("cap", 0.90, (180, 80, 220, 130))
    LAB = ("label", 0.85, (155, 250, 245, 420))
    good, nocap, nolabel = det(BOT, CAP, LAB), det(BOT, LAB), det(BOT, CAP)

    # clutter is not a bottle: the boxes seen on the real belt with NO bottle in view (low confidence, square or
    # wide shapes) must be dropped, and a real bottle with its parts kept
    R = dict(RULES)
    wide = det(("bottle", 0.41, (0, 0, 560, 330)), ("cap", 0.39, (10, 0, 80, 50)), wh=(1920, 1080))
    square = det(("bottle", 0.62, (0, 100, 300, 400)), wh=(1920, 1080))
    unsure = det(("bottle", 0.41, (700, 100, 900, 600)), wh=(1920, 1080))          # tall but not confident
    assert gate_detections(wide, R) == () and gate_detections(square, R) == () and gate_detections(unsure, R) == ()
    assert [d.class_name for d in gate_detections(det(BOT, CAP, LAB), R)] == ["bottle", "cap", "label"]
    assert [d.class_name for d in gate_detections(det(BOT, ("label", 0.2, (155, 250, 245, 420))), R)] == ["bottle"]
    assert detection_findings(wide, R) == (None, None)                              # "no bottle", never a defect
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
    # defect localisation: a recipe's defect class inside the bottle IS the defect, located at its box
    names4 = tuple(DET_CLASSES) + ("cap_tilted", "label_damaged")
    dr = {**RULES, "recipe": {**default_recipe(RULES),
                              "defects": {"cap_tilted": {"finding": "tilt_cap", "min_conf": 0.6}, "label_damaged": {}}}}
    TILT = ("cap_tilted", 0.8, (178, 78, 222, 132))
    tilted = det(BOT, CAP, LAB, TILT, names=names4)
    f, c = detection_findings(tilted, dr)
    assert f == ["tilt_cap"] and abs(c - 0.95) < 1e-9, (f, c)
    loc = defect_locations(tilted, dr)
    assert loc == [("tilt_cap", 0.8, (178, 78, 222, 132))], loc
    assert detection_findings(det(BOT, CAP, LAB, ("cap_tilted", 0.5, TILT[2]), names=names4), dr)[0] == []   # < min_conf
    assert detection_findings(det(BOT, CAP, LAB, ("cap_tilted", 0.9, (600, 80, 640, 130)), names=names4), dr)[0] == []  # outside
    assert recipe_classes(dr["recipe"]) == ("bottle", "cap", "label", "cap_tilted", "label_damaged")
    assert recipe_problems(dr["recipe"]) == []
    assert recipe_problems({**dr["recipe"], "defects": {"cap": {}}})                    # clashes with a part
    assert recipe_problems({**dr["recipe"], "defects": {"x": {"min_conf": 1.5}}})
    # a missing part is located where it should have been (the cap zone at the top of the bottle)
    (name, _, (x1, y1, x2, y2)), = defect_locations(nocap, RULES)
    assert name == "missing_cap" and (x1, x2) == (150, 250) and y1 < 100 < y2 < 330, (name, x1, y1, x2, y2)
    # the bottle's decision carries the locations of the defects it reports (best box per camera and finding)
    dd = decide("detection", [cam({"det": tilted}, {"det": tilted})], {}, {"recipe": dr["recipe"]})
    assert dd.state == REJECT and dd.defects == ["tilt_cap"] and dd.locations == [("cam0", 1, "tilt_cap", 0.8,
                                                                                  (178.0, 78.0, 222.0, 132.0))], dd
    assert D("detection", cam({"det": good})).locations == []
    # any_of: a v4 detector labels a tilted cap ONLY as cap_tilted. The part accepts it (not "missing_cap") and the
    # defects section reports the tilt, located at the cap
    v4 = {**RULES, "recipe": {"anchor": "bottle",
                              "parts": [{**default_recipe(RULES)["parts"][0], "any_of": ["cap_tilted"]},
                                        {"name": "label", "search": [0.0, 1.0]}],
                              "defects": {"cap_tilted": {"finding": "tilt_cap"}}}}
    only_tilt = det(BOT, LAB, TILT, names=names4)
    assert detection_findings(only_tilt, v4)[0] == ["tilt_cap"], detection_findings(only_tilt, v4)
    assert detection_findings(only_tilt, dr)[0] == ["missing_cap", "tilt_cap"]       # without any_of: also "missing"
    assert recipe_classes(v4["recipe"]) == ("bottle", "cap", "cap_tilted", "label")
    assert recipe_problems(v4["recipe"]) == [] and recipe_problems(
        {"anchor": "bottle", "parts": [{"name": "cap", "any_of": "cap_tilted"}]})
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
    # three-zone rule: PASS needs every score <= pass_max; between pass_max and the threshold is UNSURE -> FAULT
    Z = {"pass_max": 0.15}
    mid_p = {"tilt_cap": 0.3, "missing_label": 0.05}                                 # 0.3: under 0.5, over 0.15
    assert D("classification", cam({"probs": mid_p}, {"probs": mid_p}), thr=thr).state == PASS   # old rule passes it
    d = D("classification", cam({"probs": mid_p}, {"probs": mid_p}), thr=thr, rules=Z)
    assert d.state == FAULT and "unsure: tilt_cap" in d.reason, d
    low_p = {"tilt_cap": 0.1, "missing_label": 0.12}
    assert D("classification", cam({"probs": low_p}, {"probs": low_p}), thr=thr, rules=Z).state == PASS
    assert D("classification", cam({"probs": bad_p}, {"probs": bad_p}), thr=thr, rules=Z).defects == ["tilt_cap"]
    # a confident defect still rejects even when another score is unsure (REJECT is the answer, not FAULT)
    both = {"tilt_cap": 0.9, "missing_label": 0.3}
    assert D("classification", cam({"probs": both}, {"probs": both}), thr=thr, rules=Z).state == REJECT
    # a disabled defect (threshold 1.01) is never "unsure", whatever it scores
    assert D("classification", cam({"probs": {"tilt_cap": 0.1, "missing_cap": 0.8}}),
             thr={**thr, "missing_cap": 1.01}, rules=Z).state == PASS
    # one unsure frame of three is outvoted (majority), two of three are not
    assert D("classification", cam({"probs": mid_p}, {"probs": low_p}, {"probs": low_p}), thr=thr, rules=Z).state == PASS
    assert D("classification", cam({"probs": mid_p}, {"probs": mid_p}, {"probs": low_p}), thr=thr, rules=Z).state == FAULT
    # minimum share of frames that must see the bottle
    F = {"min_bottle_frames_frac": 0.6}
    assert D("detection", cam({"det": good}, {"det": empty}, {"det": empty})).state == PASS       # off: 1 of 3 is enough
    d = D("detection", cam({"det": good}, {"det": empty}, {"det": empty}), rules=F)
    assert d.state == FAULT and "only 1 of 3" in d.reason, d
    assert D("detection", cam({"det": good}, {"det": good}, {"det": empty}), rules=F).state == PASS
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
