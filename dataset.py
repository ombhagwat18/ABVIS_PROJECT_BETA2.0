"""labels.csv is the source of truth. Folders are just where the JPGs live.

One row per image, one column per defect. GOOD = every column is 0, so "good"
can never contradict a defect flag. Adding a defect = adding a column.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECTS = ROOT / "projects"
ACTIVE_TXT = PROJECTS / "active.txt"

POS_DIR, NEG_DIR, INBOX_DIR = "+ve", "-ve", "_inbox"

# The AI task a project trains/serves. Only "classification" is implemented
# today (this is what every existing project -- including om_bottle -- runs);
# "detection" and "segmentation" are accepted values reserved for future
# stages and are not yet wired into train.py/infer.py.
TASKS = ("classification", "detection", "segmentation")
DEFAULT_TASK = "classification"

# Rebound by use_project(). Every other module reads these as D.NAME, which is
# an attribute lookup on this module at call time, so switching project needs no
# change anywhere else -- gui, train, infer and calibrate all follow.
PROJECT = ""
TASK = DEFAULT_TASK
PROJECT_DIR = IMAGE_ROOT = POS = NEG = INBOX = DATA = Path()
LABELS_CSV = CONFIG_JSON = SCENES_JSON = THUMBS = CACHE = MODELS = Path()

# A bottle is tall and narrow. Squeezing that ROI into a square leaves the
# bottle ~142 px wide in a 320 px frame with more than half the pixels black --
# resolution spent on nothing, on the exact detail (meniscus line, label print)
# the classes turn on. Keep the input tall instead.
INPUT_WH = (192, 448)             # what the network sees
CACHE_WH = (216, 496)             # cached slightly larger so augment can crop
THUMB_SIZE = 200

DEFAULT_CONFIG = {
    # [x, y, w, h] in original pixels. null = whole frame. Fixed camera means a
    # fixed ROI is correct; run `python calibrate.py` to measure it.
    "roi": None,
    "roi_frame": None,                # [W, H] the ROI was measured on; lets it rescale
    "input_wh": list(INPUT_WH),
    "cache_wh": list(CACHE_WH),       # see cache_wh(); must be >= input_wh
    "camera": 0,
    "thresholds": {},
}

# ---------------------------------------------------------------------- project

def use_project(name: str) -> str:
    """Point every path at projects/<name>/ and remember the choice.

    Rebinding module globals rather than threading a project object through the
    call graph is deliberate: the other four modules already say D.MODELS and
    D.IMAGE_ROOT, which resolve on this module at call time. One assignment here
    moves all of them.
    """
    global PROJECT, TASK, PROJECT_DIR, IMAGE_ROOT, POS, NEG, INBOX, DATA
    global LABELS_CSV, CONFIG_JSON, SCENES_JSON, THUMBS, CACHE, MODELS, _cfg_cache
    PROJECT = name
    TASK = project_task(name)
    PROJECT_DIR = PROJECTS / name
    IMAGE_ROOT = PROJECT_DIR / "images"
    POS, NEG, INBOX = IMAGE_ROOT / POS_DIR, IMAGE_ROOT / NEG_DIR, IMAGE_ROOT / INBOX_DIR
    DATA = PROJECT_DIR
    LABELS_CSV = DATA / "labels.csv"
    CONFIG_JSON = DATA / "config.json"
    SCENES_JSON = DATA / "scenes.json"
    THUMBS, CACHE = DATA / "thumbs", DATA / "cache"
    MODELS = PROJECT_DIR / "models"
    _cfg_cache = None                  # the cache is keyed on mtime, not on path
    PROJECTS.mkdir(parents=True, exist_ok=True)
    ACTIVE_TXT.write_text(name, encoding="utf-8")
    return name


def list_projects() -> list[str]:
    if not PROJECTS.is_dir():
        return []
    return sorted(p.name for p in PROJECTS.iterdir() if (p / "images").is_dir())


def project_title(name: str) -> str:
    """The name the admin typed; the folder is a slug of it."""
    try:
        return json.loads((PROJECTS / name / "project.json").read_text())["title"]
    except Exception:
        return name


def project_task(name: str) -> str:
    """The AI task a project runs. Missing field or missing/unreadable file ->
    "classification" -- every project that existed before this field was
    introduced (including om_bottle) must keep behaving exactly as it does
    today, with no migration step required."""
    try:
        task = json.loads((PROJECTS / name / "project.json").read_text()).get("task")
    except Exception:
        return DEFAULT_TASK
    return task if task in TASKS else DEFAULT_TASK


def create_project(title: str, task: str = DEFAULT_TASK) -> str:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r} -- choices: {', '.join(TASKS)}")
    name = slug(title)
    if not name:
        raise ValueError("give the project a name")
    if (PROJECTS / name).exists():
        raise ValueError(f"project {name!r} already exists")
    for sub in (f"images/{POS_DIR}", f"images/{NEG_DIR}", f"images/{INBOX_DIR}", "models"):
        (PROJECTS / name / sub).mkdir(parents=True)
    (PROJECTS / name / "project.json").write_text(
        json.dumps({"title": title.strip(), "task": task}, indent=2), encoding="utf-8")
    return name


def _boot() -> None:
    """Open the remembered project, else the first one, else an empty default."""
    want = ACTIVE_TXT.read_text(encoding="utf-8").strip() if ACTIVE_TXT.exists() else ""
    have = list_projects()
    if want not in have:
        want = have[0] if have else create_project("Default")
    use_project(want)


# --------------------------------------------------------------------------- io

def _natkey(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


_cfg_cache: tuple[float, dict] | None = None


def load_config() -> dict:
    """Re-read only when the file actually changes. The inference loop asks for
    the thresholds on every frame; without this it is ~60 disk reads a second."""
    global _cfg_cache
    mtime = CONFIG_JSON.stat().st_mtime if CONFIG_JSON.exists() else 0.0
    if _cfg_cache and _cfg_cache[0] == mtime:
        return dict(_cfg_cache[1])
    cfg = dict(DEFAULT_CONFIG)
    if mtime:
        cfg.update(json.loads(CONFIG_JSON.read_text()))
    _cfg_cache = (mtime, dict(cfg))
    return cfg


def save_config(cfg: dict) -> None:
    global _cfg_cache
    DATA.mkdir(parents=True, exist_ok=True)
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2))
    _cfg_cache = None


# ----------------------------------------------------------- app-wide settings
# Not per project: font size and how far to probe for cameras belong to this
# installation, and following the project switcher around would be surprising.

SETTINGS_JSON = ROOT / "settings.json"

DEFAULT_SETTINGS = {
    "font_scale": 1.0,           # 0.8 - 1.6, applied to widget and text scaling
    "grid_columns": 0,           # 0 = fit to the window
    "per_page": 60,
    "camera_probe": 5,           # how many indices list_cameras() walks
    "bench_seconds": 3.0,
    "monitor_hz": 2,             # performance panel refresh rate
}


def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    try:
        s.update(json.loads(SETTINGS_JSON.read_text()))
    except Exception:
        pass                      # a corrupt settings file must not stop the app
    return s


def save_settings(s: dict) -> None:
    SETTINGS_JSON.write_text(json.dumps(s, indent=2))


def imread(path: Path) -> np.ndarray | None:
    """cv2.imread chokes on some Windows paths; go through numpy."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------------------------------------------------------------------- labels

def scan_images() -> list[str]:
    """Every JPG under IMAGE_ROOT, as a forward-slash path relative to it."""
    out = []
    for p in IMAGE_ROOT.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file():
            out.append(p.relative_to(IMAGE_ROOT).as_posix())
    return sorted(out, key=_natkey)


def scan_classes() -> list[str]:
    """Columns implied by the -ve/ sub-folders, including still-empty ones.

    An empty folder still earns its column so the admin can make the class and
    then upload into it. train.py gives any defect with no training images an
    unreachable threshold, so an empty column cannot fire.
    """
    if not NEG.is_dir():
        return []
    return sorted({slug(p.name) for p in NEG.iterdir() if p.is_dir() and slug(p.name)})


def implied(rel: str) -> tuple[str | None, int]:
    """(column to set, reviewed) that a file's position in the tree asserts."""
    parts = rel.split("/")
    if parts[0] == NEG_DIR:
        # -ve/<Class>/file.jpg. A file dropped straight into -ve/ names no class,
        # and treating it as good would teach the model that a reject passes --
        # park it in the inbox for someone to classify instead.
        return (slug(parts[1]), 1) if len(parts) > 2 else (None, 0)
    if parts[0] == POS_DIR:
        return None, 1                 # good: every column stays 0
    return None, 0                     # _inbox/ and anything else: undecided


# Not a defect: distinguishes "reviewed, and it is a good bottle" (all zeros,
# reviewed=1) from "nobody has looked at this yet" (all zeros, reviewed=0).
# Without it those two are the same row and the inbox is impossible.
RESERVED = ("reviewed",)


def load_labels() -> tuple[list[str], dict[str, dict[str, int]]]:
    """-> (defect names in column order, {relpath: {defect: 0|1, 'reviewed': 0|1}})"""
    if not LABELS_CSV.exists():
        return import_from_folders()
    with LABELS_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return import_from_folders()
    cols = rows[0][1:]
    defects = [c for c in cols if c not in RESERVED]
    labels = {}
    for r in rows[1:]:
        if not r:
            continue
        row = {c: int(v or 0) for c, v in zip(cols, r[1:])}
        row.setdefault("reviewed", 1)
        labels[r[0]] = row
    return defects, sync_new_images(defects, labels)


def sync_new_images(defects, labels) -> dict:
    """Fold the folder tree into the CSV: new classes, new files, gone files.

    A NEW file gets the label its folder implies. A file already in the CSV is
    left alone, and that is the whole rule: a folder can only ever say one defect
    per image, but the model is multi-label. This project has already paid for
    the difference -- 19 Water Level frames were also visibly tilt_cap, the
    folder could not record both, and it cost real precision. Second defects get
    ticked in the Label tab, and a rescan must not wipe them.

    `defects` is mutated in place, so callers holding the list see new columns.
    """
    on_disk = set(scan_images())
    changed = False

    def column(d: str) -> None:
        if d not in defects and d not in RESERVED:
            defects.append(d)
            for row in labels.values():
                row.setdefault(d, 0)

    before = len(defects)
    for d in scan_classes():
        column(d)
    for p in on_disk - labels.keys():
        d, reviewed = implied(p)
        if d:
            column(d)
        labels[p] = {**{x: 0 for x in defects}, "reviewed": reviewed}
        if d:
            labels[p][d] = 1
        changed = True
    for p in labels.keys() - on_disk:
        del labels[p]
        changed = True

    defects.sort()
    if changed or len(defects) != before:
        save_labels(defects, labels)
    return labels


def save_labels(defects: list[str], labels: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = LABELS_CSV.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", *defects, *RESERVED])
        for p in sorted(labels, key=_natkey):
            row = labels[p]
            w.writerow([p, *(row.get(d, 0) for d in defects), *(row.get(c, 1) for c in RESERVED)])
    tmp.replace(LABELS_CSV)


def import_from_folders() -> tuple[list[str], dict]:
    """First run for a project: build labels.csv from the +ve / -ve tree."""
    defects: list[str] = []
    labels: dict = {}
    sync_new_images(defects, labels)
    if not LABELS_CSV.exists():
        save_labels(defects, labels)     # an empty project still gets a header
    return defects, labels


def is_good(row, defects) -> bool:
    return not any(row.get(d, 0) for d in defects)


# ---------------------------------------------------------------- operations
# The GUI and the HTTP API both drive these. Keeping them here rather than in
# either front end means "what a label edit does" has exactly one definition.

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def apply_labels(paths, defect=None, value=1, clear_all=False) -> dict:
    """Set or clear a defect on many images. Always one CSV rewrite, whatever
    the edit -- 'mark good' must not cost one full rewrite per defect column."""
    defects, labels = load_labels()
    if defect is not None and defect not in defects:
        raise KeyError(f"unknown defect {defect!r}")
    for p in paths:
        if p not in labels:
            continue
        if clear_all:
            for d in defects:
                labels[p][d] = 0
        elif defect is not None:
            labels[p][defect] = int(value)
        labels[p]["reviewed"] = 1
    save_labels(defects, labels)
    return counts(defects, labels)


def _free(target: Path) -> Path:
    """A path that is not taken yet -- snap_001.jpg exists in several folders."""
    out, i = target, 1
    while out.exists():
        out = target.with_name(f"{target.stem}_{i}{target.suffix}")
        i += 1
    return out


def _class_dir(defect: str) -> Path | None:
    """The -ve/ folder whose name slugs to this column, if one exists."""
    if not NEG.is_dir():
        return None
    return next((p for p in NEG.iterdir() if p.is_dir() and slug(p.name) == defect), None)


def add_defect(name: str) -> list[str]:
    defects, labels = load_labels()
    n = slug(name)
    if not n:
        raise ValueError("empty name")
    if n in defects or n in RESERVED:
        raise ValueError(f"{n!r} already exists")
    (NEG / n).mkdir(parents=True, exist_ok=True)   # so Upload has a destination
    defects.append(n)
    for row in labels.values():
        row[n] = 0
    save_labels(sorted(defects), labels)
    return sorted(defects)


def rename_defect(old: str, new: str) -> list[str]:
    defects, labels = load_labels()
    n = slug(new)
    if old not in defects:
        raise KeyError("no such defect")
    if not n or n in defects or n in RESERVED:
        raise ValueError("bad new name")
    folder = _class_dir(old)
    if folder:
        # The folder has to move too. Left behind, the next scan would read it as
        # a class that still exists and resurrect the column just renamed away.
        # Rewrite the rows to the new paths rather than re-importing the tree --
        # a re-import would reset every image back to its one folder-implied
        # defect and throw away the second defects someone ticked by hand.
        new_dir = _free(NEG / n)
        folder.rename(new_dir)
        was, now = f"{NEG_DIR}/{folder.name}/", f"{NEG_DIR}/{new_dir.name}/"
        labels = {(now + k[len(was):] if k.startswith(was) else k): v
                  for k, v in labels.items()}
    defects[defects.index(old)] = n
    for row in labels.values():
        row[n] = row.pop(old, 0)
    save_labels(sorted(defects), labels)
    return sorted(defects)


def delete_defect(name: str) -> list[str]:
    """Drop the column, and send its images to the inbox rather than to good.

    The -ve/ folder goes too, or the next scan rebuilds the column. Its images
    are not good bottles, they are undecided, so they land in _inbox/ unreviewed
    for someone to reclassify -- silently marking a known reject as a pass is
    the one outcome that must not happen here.
    """
    defects, _ = load_labels()
    if name not in defects:
        raise KeyError("no such defect")
    folder = _class_dir(name)
    if folder:
        INBOX.mkdir(parents=True, exist_ok=True)
        for p in folder.iterdir():
            if p.is_file():
                p.rename(_free(INBOX / p.name))
        shutil.rmtree(folder, ignore_errors=True)
    defects, labels = load_labels()        # picks up the moved files as inbox
    defects.remove(name)
    for row in labels.values():
        row.pop(name, None)
    save_labels(sorted(defects), labels)
    return sorted(defects)


def add_images(sources, dest: str) -> int:
    """Copy files into the tree. `dest` is relative to images/, e.g. '+ve'.

    Copy, not move: the admin picked these out of a camera dump or a share, and
    a failed import that also emptied the source folder is not recoverable.
    """
    out = IMAGE_ROOT / dest
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sources:
        src = Path(src)
        if src.suffix.lower() not in (".jpg", ".jpeg", ".png") or not src.is_file():
            continue
        shutil.copy2(src, _free(out / src.name))
        n += 1
    return n


def add_folder(src_dir, dest: str) -> int:
    """Same, for every image in a folder the admin points at (one level deep)."""
    src_dir = Path(src_dir)
    return add_images(sorted(p for p in src_dir.rglob("*") if p.is_file()), dest)


def delete_images(paths) -> dict:
    """Remove images from disk and from labels.csv.

    Only touches files under IMAGE_ROOT -- a path that escapes it (via .. or an
    absolute path) is refused rather than followed.
    """
    root = IMAGE_ROOT.resolve()
    defects, labels = load_labels()
    for rel in paths:
        p = (IMAGE_ROOT / rel).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            raise ValueError(f"refusing to delete outside the dataset: {rel}")
        p.unlink(missing_ok=True)
        labels.pop(rel, None)
        for cache_dir in (THUMBS, CACHE):
            for stale in cache_dir.rglob(rel.replace("/", "__") + ".*"):
                stale.unlink(missing_ok=True)
    save_labels(defects, labels)
    if SCENES_JSON.exists():
        SCENES_JSON.unlink()            # the grouping is keyed on the path set
    return counts(defects, labels)


def save_capture(frame_bgr: np.ndarray, defects_on=(), reviewed=True) -> str:
    """Write a live frame into the dataset, already labelled."""
    INBOX.mkdir(parents=True, exist_ok=True)
    rel = f"{INBOX_DIR}/cap_{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
    cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
        str(IMAGE_ROOT / rel))
    defects, labels = load_labels()          # sync_new_images picks the file up
    wanted = [d for d in defects_on if d in defects]
    labels.setdefault(rel, {d: 0 for d in defects})
    for d in defects:
        labels[rel][d] = 1 if d in wanted else 0
    labels[rel]["reviewed"] = 1 if reviewed else 0
    save_labels(defects, labels)
    return rel


def counts(defects, labels) -> dict:
    """Per-defect counts, plus the pass/fail split of the whole dataset.

    Two different things get called "positive" around here, so neither is:
    `_good` / `_defective` is the dataset-level split (a bottle passes or it
    does not), while c[defect] is how many images carry that one defect. An
    image with two defects is one defective bottle but counts under both.
    """
    c = {d: sum(r.get(d, 0) for r in labels.values()) for d in defects}
    c["_good"] = sum(1 for r in labels.values() if r.get("reviewed", 1) and is_good(r, defects))
    c["_defective"] = sum(1 for r in labels.values() if not is_good(r, defects))
    c["_unreviewed"] = sum(1 for r in labels.values() if not r.get("reviewed", 1))
    c["_total"] = len(labels)
    return c


# ------------------------------------------------------------------ crop/cache

def crop(img: np.ndarray, roi, roi_frame=None) -> np.ndarray:
    """Crop to the ROI, rescaling it if this frame is a different size.

    The ROI is measured in absolute pixels on the training images. A camera or a
    video at another resolution is the same scene at another scale, so the box
    has to scale with it -- otherwise a half-size frame crops the right-hand edge
    of the picture and the model confidently reads background.
    """
    if not roi:
        return img
    x, y, w, h = roi
    H, W = img.shape[:2]
    if roi_frame and roi_frame[0] and roi_frame[1] and (roi_frame[0], roi_frame[1]) != (W, H):
        sx, sy = W / roi_frame[0], H / roi_frame[1]
        x, y, w, h = x * sx, y * sy, w * sx, h * sy
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, x0 + int(w)), min(H, y0 + int(h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return img
    return img[y0:y1, x0:x1]


def letterbox(img: np.ndarray, wh) -> np.ndarray:
    """Resize keeping aspect, pad into a (w, h) box. Aspect is preserved, never
    squashed: stretching a tall bottle changes the cap angle and the meniscus
    slope, which is what two of the classes are actually about."""
    tw, th = (wh, wh) if isinstance(wh, int) else wh
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((th, tw, 3), np.uint8)
    top, left = (th - nh) // 2, (tw - nw) // 2
    out[top:top + nh, left:left + nw] = r
    return out


def input_wh(cfg: dict) -> tuple[int, int]:
    return tuple(cfg.get("input_wh") or INPUT_WH)


def cache_wh(cfg: dict) -> tuple[int, int]:
    """The cached crop size: a little larger than the input, so augmentation has
    room to crop inside it.

    Per project, not a constant. A project whose object is not a tall bottle --
    a carton at 448x192 -- would otherwise centre-crop a 216-wide cache to 448
    and get a short frame back, silently, with no error and a model reading
    padding. The default is exactly the (216, 496) it always was, so every
    existing checkpoint sees the identical framing.
    """
    w, h = input_wh(cfg)
    cw, ch = cfg.get("cache_wh") or (0, 0)
    return (cw, ch) if cw >= w and ch >= h else (round(w * 1.125), round(h * 1.107))


def prepare(img: np.ndarray, cfg: dict, wh=None) -> np.ndarray:
    return letterbox(crop(img, cfg.get("roi"), cfg.get("roi_frame")), wh or input_wh(cfg))


def center_crop(img: np.ndarray, wh) -> np.ndarray:
    tw, th = wh
    h, w = img.shape[:2]
    oy, ox = max(0, (h - th) // 2), max(0, (w - tw) // 2)
    return img[oy:oy + th, ox:ox + tw]


def model_input(full_bgr: np.ndarray, cfg: dict, wh=None) -> np.ndarray:
    """The exact view the model is scored on. Inference MUST go through this.

    Training reads the cached CACHE_WH crop and centre-crops it to the input
    size, so the network is validated on a slightly tighter framing than a
    straight letterbox produces. Feeding it the untightened view at inference
    is train/serve skew: it moves the meniscus a couple of percent up the frame,
    which is most of what water_level has to go on.
    """
    return center_crop(letterbox(crop(full_bgr, cfg.get("roi"), cfg.get("roi_frame")), cache_wh(cfg)),
                       wh or input_wh(cfg))


def _cache_dir(cfg: dict) -> Path:
    key = hashlib.md5(json.dumps([cfg.get("roi"), cache_wh(cfg), "png"]).encode()).hexdigest()[:8]
    return CACHE / key


def cached_crop(rel: str, cfg: dict) -> np.ndarray | None:
    """Decoding 2537x1927 JPEGs every epoch costs ~50 ms each. Cache the crop
    once; every epoch after the first reads a small file instead.

    PNG, not JPEG. A cache is an optimisation and must not change the data:
    re-encoding at quality 95 moved pixels by ~2 grey levels, which was enough
    to swing a borderline water_level probability by 0.15 and made validation
    disagree with live inference on 15 of 31 frames.
    """
    cp = _cache_dir(cfg) / (rel.replace("/", "__") + ".png")
    if cp.exists():
        return imread(cp)
    img = imread(IMAGE_ROOT / rel)
    if img is None:
        return None
    out = prepare(img, cfg, cache_wh(cfg))
    cp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(str(cp))
    return out


def thumbnail(rel: str) -> bytes | None:
    tp = THUMBS / (rel.replace("/", "__") + ".jpg")
    if tp.exists():
        return tp.read_bytes()
    img = imread(IMAGE_ROOT / rel)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = THUMB_SIZE / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_bytes(buf)
    return buf


# ------------------------------------------------------------------- the split

def scene_map(paths: list[str], threshold=4.0) -> dict[str, str]:
    """Group consecutive near-identical frames into one scene.

    These images are video frames: 1145 of them are about 112 actual bottles,
    in runs of 18-31 nearly identical shots. A fixed-size block does not line up
    with those runs, so the same bottle still lands on both sides of the split
    and validation scores come back at a meaningless 1.000. Detect the runs and
    keep each one whole instead. Cached, because it decodes every thumbnail.
    """
    key = hashlib.md5(("|".join(sorted(paths)) + f"|{threshold}").encode()).hexdigest()
    if SCENES_JSON.exists():
        try:
            c = json.loads(SCENES_JSON.read_text())
            if c.get("key") == key:
                return c["map"]
        except Exception:
            pass

    by_folder: dict[str, list[str]] = {}
    for p in paths:
        by_folder.setdefault(p.rsplit("/", 1)[0] if "/" in p else "", []).append(p)

    out: dict[str, str] = {}
    for folder, items in sorted(by_folder.items()):
        items.sort(key=_natkey)
        prev, sid = None, 0
        for p in items:
            b = thumbnail(p)
            sig = None
            if b is not None:
                a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
                if a is not None:
                    sig = cv2.resize(a, (32, 32)).astype(np.float32)
            if sig is None or prev is None or float(np.abs(sig - prev).mean()) >= threshold:
                sid += 1          # unreadable frames become their own scene
            out[p] = f"{folder}#{sid}"
            prev = sig

    DATA.mkdir(parents=True, exist_ok=True)
    SCENES_JSON.write_text(json.dumps({"key": key, "map": out}))
    return out


def split_train_val(paths: list[str], val_frac=0.2, seed=0, groups=None):
    """Split by SCENE, never at random.

    snap_1000..snap_1089 are consecutive frames of one bottle and differ by a
    few hundred bytes. A random split puts frame 1042 in train and 1043 in val,
    and validation recall becomes a number about JPEG noise, not about defects.
    A whole scene goes to one side or the other.
    """
    groups = groups if groups is not None else scene_map(paths)
    by_folder: dict[str, dict[str, list[str]]] = {}
    for p in paths:
        folder = p.rsplit("/", 1)[0] if "/" in p else ""
        by_folder.setdefault(folder, {}).setdefault(groups.get(p, p), []).append(p)

    train, val = [], []
    for folder, scenes in sorted(by_folder.items()):
        keys = sorted(scenes, key=_natkey)
        n_items = sum(len(scenes[k]) for k in keys)
        # A quota, not an independent coin flip per scene. Flipping per scene
        # means a 2-scene class lands entirely in train most of the time and
        # then reports a meaningless 0.000 recall on zero positives.
        order = sorted(keys, key=lambda k: hashlib.md5(f"{seed}:{k}".encode()).hexdigest())
        target = max(1, round(n_items * val_frac))
        chosen, n = set(), 0
        for k in order:
            if n >= target or len(chosen) >= len(keys) - 1:
                break  # validation never swallows the whole folder
            chosen.add(k)
            n += len(scenes[k])
        for k in keys:
            (val if k in chosen else train).extend(sorted(scenes[k], key=_natkey))
    return train, val


def split_train_val_test(paths: list[str], val_frac=0.2, test_frac=0.15, seed=0, groups=None):
    """A genuinely held-out test set, on top of the existing scene split.

    Composed from two calls to split_train_val instead of new grouping logic:
    the first carves TEST out of everything, the second carves VAL out of
    whatever is left. Both calls share the same scene groups but each only
    ever sees the paths still in its own pool, so a scene already sent to
    test in call 1 cannot resurface in train or val from call 2 -- the
    disjointness guarantee is inherited from split_train_val's own (already
    verified) property, not re-implemented. A different seed for each call
    keeps the test carve-out independent of the train/val boundary.
    """
    groups = groups if groups is not None else scene_map(paths)
    keep, test = split_train_val(paths, val_frac=test_frac, seed=seed + 90000, groups=groups)
    # val_frac was a fraction of ALL paths; re-express it as a fraction of what
    # remains after test is removed, so the overall proportions still land
    # close to what the caller asked for.
    remaining_val_frac = val_frac / max(1e-6, 1 - test_frac)
    train, val = split_train_val(keep, val_frac=min(0.9, remaining_val_frac), seed=seed, groups=groups)
    return train, val, test


def demo():
    """Self-check: the split must never put one scene on both sides."""
    def scenes(folder, n, per):        # n frames of the same bottle, per scene
        ps = [f"{folder}/snap_{i}.jpg" for i in range(n)]
        return ps, {p: f"{folder}#{i // per}" for i, p in enumerate(ps)}

    pa, ga = scenes("A", 300, 20)
    pb, gb = scenes("B", 120, 20)
    paths, groups = pa + pb, {**ga, **gb}
    tr, va = split_train_val(paths, val_frac=0.2, groups=groups)
    assert set(tr) | set(va) == set(paths), "every image must land somewhere"
    assert not (set(tr) & set(va)), "no image in both halves"
    assert va, "validation must be non-empty"

    # THE property this whole function exists for: no scene spans the split.
    tset = set(tr)
    for g in set(groups.values()):
        members = [p for p in paths if groups[p] == g]
        assert len({p in tset for p in members}) == 1, f"scene {g} leaked across the split"

    # Every folder must get validation however small, and training keeps the
    # bulk. A 30-image class returning 0.000 recall on 0 positives is not a bad
    # score, it is no score, and it reads as a bad one.
    for n in (10, 30, 51, 72, 120, 300):
        p, g = scenes("S", n, max(1, n // 6))
        t, v = split_train_val(p, val_frac=0.2, groups=g)
        assert v, f"{n} images produced an empty validation set"
        assert t, f"{n} images produced an empty training set"
        assert len(t) + len(v) == n
        assert len(t) > len(v), f"{n} images put more in validation ({len(v)}) than train ({len(t)})"

    p, g = scenes("T", 4, 4)           # one scene cannot be split
    assert not split_train_val(p, groups=g)[1]
    # folders split independently, so one huge folder cannot starve a small one
    pb2, gb2 = scenes("big", 400, 20)
    pt, gt = scenes("tiny", 30, 6)
    _, v = split_train_val(pb2 + pt, groups={**gb2, **gt})
    assert any(x.startswith("tiny/") for x in v), "small folder got no validation"
    # the real property: no train frame is adjacent to a val frame
    idx = {p: i for i, p in enumerate(sorted(paths, key=_natkey))}
    vset = set(va)
    boundaries = sum(1 for p in paths if p in vset and
                     any(q not in vset for q in paths if abs(idx[q] - idx[p]) == 1))
    assert boundaries <= 2 * (len(va) // 25 + 2), "blocks are being broken up"

    # a held-out test set must be disjoint from train AND val, at the scene
    # level, on both sides -- not just "no duplicate path"
    tr3, va3, te3 = split_train_val_test(paths, val_frac=0.2, test_frac=0.15, groups=groups)
    assert set(tr3) | set(va3) | set(te3) == set(paths), "every image must land in exactly one split"
    assert not (set(tr3) & set(va3)) and not (set(tr3) & set(te3)) and not (set(va3) & set(te3)), \
        "train/val/test must not overlap"
    assert te3, "test split must be non-empty"
    for g in set(groups.values()):
        members = {p for p in paths if groups[p] == g}
        in_tr, in_va, in_te = members & set(tr3), members & set(va3), members & set(te3)
        assert sum(bool(x) for x in (in_tr, in_va, in_te)) == 1, \
            f"scene {g} spans more than one of train/val/test"

    assert crop(np.zeros((100, 100, 3), np.uint8), None).shape == (100, 100, 3)
    assert crop(np.zeros((100, 100, 3), np.uint8), [10, 10, 50, 40]).shape == (40, 50, 3)
    assert crop(np.zeros((100, 100, 3), np.uint8), [90, 90, 50, 50]).shape == (10, 10, 3)

    # A half-size frame must crop the same part of the scene, not the same
    # pixel coordinates. Without this a downscaled video silently feeds the
    # model the right-hand edge of the picture.
    big = np.zeros((1927, 2537, 3), np.uint8)
    big[0:1927, 820:1673] = 255                       # the "bottle", at the ROI
    roi, rf = [820, 0, 853, 1927], [2537, 1927]
    assert crop(big, roi, rf).mean() == 255
    half = cv2.resize(big, (1268, 963))
    assert crop(half, roi, rf).mean() > 250, "rescaled ROI missed the subject"
    assert crop(half, roi, None).mean() < 130, "unscaled ROI should miss it (the bug)"

    # a tall crop must fill a tall box, not sit in a square surrounded by black
    tall = np.full((1927, 853, 3), 255, np.uint8)
    out = letterbox(tall, INPUT_WH)
    assert out.shape == (INPUT_WH[1], INPUT_WH[0], 3), out.shape
    fill = (out > 0).any(2).mean()
    assert fill > 0.85, f"only {fill:.0%} of the input is bottle -- aspect mismatch"
    assert letterbox(tall, 64).shape == (64, 64, 3)          # int still means square
    assert (letterbox(np.full((100, 400, 3), 255, np.uint8), (80, 80))[0] == 0).all()

    # Train/serve skew guard. What validation feeds the network (cached crop,
    # centre-cropped) and what inference feeds it must be the same pixels. When
    # these drift apart the model still scores well and still fails live.
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (1927, 2537, 3), dtype=np.uint8)
    cfg = {"roi": [820, 0, 853, 1927]}
    val_view = center_crop(prepare(frame, cfg, CACHE_WH), INPUT_WH)   # training path
    live_view = model_input(frame, cfg)                               # inference path
    assert val_view.shape == (INPUT_WH[1], INPUT_WH[0], 3), val_view.shape
    assert np.array_equal(val_view, live_view), "training and inference see different pixels"

    # deletion must not be talked into leaving the dataset folder
    for escape in ("../../secrets.txt", "a/../../../boot.ini"):
        try:
            delete_images([escape])
        except ValueError:
            pass
        else:
            raise AssertionError(f"delete_images followed {escape!r} outside the dataset")

    # ...and the on-disk cache must not alter them either
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "t.png"
        cv2.imencode(".png", val_view, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(str(f))
        assert np.array_equal(imread(f), val_view), "the crop cache is not lossless"
    # The cache must never be smaller than the input it gets centre-cropped to.
    # A wide object used to silently come back short: no error, model reads pad.
    wide = {"input_wh": [448, 192]}
    cw, ch = cache_wh(wide)
    assert cw >= 448 and ch >= 192, f"cache {cw}x{ch} is smaller than the input"
    assert model_input(np.zeros((200, 500, 3), np.uint8), wide).shape == (192, 448, 3)
    assert cache_wh({}) == CACHE_WH, "default framing moved -- existing models would skew"

    print(f"ok  train={len(tr)} val={len(va)}  scenes kept whole  "
          f"input={INPUT_WH}  fill={fill:.0%}")
    project_demo()


def project_demo():
    """Projects must not see each other, and a rescan must not undo hand labels."""
    global PROJECTS, ACTIVE_TXT
    import tempfile
    was = (PROJECTS, ACTIVE_TXT, PROJECT)
    blank = np.zeros((40, 20, 3), np.uint8)
    try:
        with tempfile.TemporaryDirectory() as td:
            PROJECTS = Path(td) / "projects"
            ACTIVE_TXT = PROJECTS / "active.txt"

            def put(rel):
                p = IMAGE_ROOT / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imencode(".jpg", blank)[1].tofile(str(p))

            a = use_project(create_project("Line A"))
            put(f"{POS_DIR}/good1.jpg")
            put(f"{NEG_DIR}/Tilt cap/t1.jpg")
            put(f"{NEG_DIR}/loose.jpg")               # dropped in with no class
            put(f"{INBOX_DIR}/cap1.jpg")
            defects, labels = load_labels()
            assert defects == ["tilt_cap"], defects
            assert labels[f"{NEG_DIR}/Tilt cap/t1.jpg"]["tilt_cap"] == 1
            assert labels[f"{POS_DIR}/good1.jpg"] == {"tilt_cap": 0, "reviewed": 1}
            # an unclassified reject must land in the inbox, never in "good"
            assert labels[f"{NEG_DIR}/loose.jpg"]["reviewed"] == 0
            assert labels[f"{INBOX_DIR}/cap1.jpg"]["reviewed"] == 0
            assert counts(defects, labels)["_good"] == 1

            # THE property the folder-vs-csv rule exists for: a folder says one
            # defect, a bottle can have two, and a rescan must not undo the second.
            add_defect("water level")
            apply_labels([f"{NEG_DIR}/Tilt cap/t1.jpg"], "water_level", 1)
            put(f"{POS_DIR}/good2.jpg")
            defects, labels = load_labels()
            assert labels[f"{NEG_DIR}/Tilt cap/t1.jpg"]["water_level"] == 1, \
                "a rescan reset a hand-made label back to what the folder implies"
            assert labels[f"{POS_DIR}/good2.jpg"]["reviewed"] == 1

            put(f"{NEG_DIR}/Missing cap/m1.jpg")      # new sub-folder = new class
            defects, labels = load_labels()
            assert "missing_cap" in defects
            assert labels[f"{NEG_DIR}/Missing cap/m1.jpg"]["missing_cap"] == 1

            assert "missing_cap" not in delete_defect("missing_cap")
            defects, labels = load_labels()
            assert "missing_cap" not in defects, "the folder resurrected the column"
            assert labels[f"{INBOX_DIR}/m1.jpg"]["reviewed"] == 0, \
                "a deleted class left its images reading as good bottles"

            rename_defect("tilt_cap", "Cap tilt")
            defects, labels = load_labels()
            assert defects == ["cap_tilt", "water_level"], defects
            row = labels[f"{NEG_DIR}/cap_tilt/t1.jpg"]
            assert row["cap_tilt"] == 1 and row["water_level"] == 1, "rename lost a label"
            n_a = len(labels)

            b = use_project(create_project("Line B"))
            assert load_labels() == ([], {}), "project B can see project A's images"
            put(f"{POS_DIR}/other.jpg")
            assert len(load_labels()[1]) == 1
            use_project(a)
            assert len(load_labels()[1]) == n_a, "project A picked up B's images"
            assert sorted(list_projects()) == [a, b]

            with tempfile.TemporaryDirectory() as src:
                f = Path(src) / "up.jpg"
                cv2.imencode(".jpg", blank)[1].tofile(str(f))
                assert add_images([f], POS_DIR) == 1
                assert add_images([f], POS_DIR) == 1      # same name, not a clobber
                assert f.exists(), "upload moved the source instead of copying it"
            assert len(load_labels()[1]) == n_a + 2
    finally:
        PROJECTS, ACTIVE_TXT = was[0], was[1]
        if was[2]:
            use_project(was[2])
    print("ok  projects isolated, folder->csv correct, hand labels survive rescan")


_boot()          # importing this module always leaves a project bound


if __name__ == "__main__":
    demo()
