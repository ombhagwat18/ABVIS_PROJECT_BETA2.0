"""Archive old models into folders named after the DATA they were trained on, and bring one back when needed.

    python -m tools.archive_models                     # plan only: what would move where (default, writes nothing)
    python -m tools.archive_models --run --reason pre-v4   # move them, write ARCHIVE_INFO.json
    python -m tools.archive_models --restore <archive folder> <name>   # copy one model back (archive keeps its copy)
    python -m tools.archive_models --selftest

Folder name:  models_archive/<YYYY-MM-DD>_cls-<labels sha8>_det-<detection data version>_<reason>/
  cls-<sha8>  first 8 hex of the labels.csv the classifiers were trained on (each checkpoint keeps a copy)
  det-<ver>   the Stage 2 split versions of the archived detector runs (v1 / v2 / v3 ...)

Rules (learned the hard way, 2026-10-13: the last manual archive moved the ACTIVE classifier and detector away and
the line could no longer load either):
  * the ACTIVE classifier (config.json active_model) and the ACTIVE detector (settings.json detector_weights, else
    models/stage2_yolo/stage2_best.pt + MODEL_PROVENANCE.json) are NEVER moved;
  * nothing is deleted; --restore copies, so the archive stays complete;
  * every move is listed in ARCHIVE_INFO.json inside the archive folder.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8] if path.is_file() else "nolabels"


def _det_version(name: str) -> str:
    m = re.search(r"_(v\d+)$", name)
    return m.group(1) if m else "v1"


def plan(root: Path, project: str, keep: tuple = (), reason: str = "archive", today: str | None = None) -> dict:
    """What would move. keep: extra names never to move. Returns {"dest", "classifiers", "detectors", "kept"}."""
    cfg_p = root / "projects" / project / "config.json"
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    st_p = root / "settings.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    active_cls = cfg.get("active_model")
    det_w = st.get("detector_weights")
    active_det_dir = Path(det_w).resolve().parent.parent if det_w else None    # .../<run>/weights/best.pt -> <run>
    kept, cls, det = [], [], []
    for d in sorted((root / "projects" / project / "models").glob("*/model.pt")):
        name = d.parent.name
        (kept if name == active_cls or name in keep else cls).append(name)
    yolo = root / "models" / "stage2_yolo"
    for d in sorted(p for p in yolo.glob("*") if p.is_dir() and (p / "weights").is_dir()):
        if (active_det_dir is not None and d.resolve() == active_det_dir) or d.name in keep:
            kept.append(d.name)
        else:
            det.append(d.name)
    if (yolo / "stage2_best.pt").exists():
        kept.append("stage2_best.pt" + ("" if det_w else " (default active detector)"))
    shas = collections.Counter(_sha8(root / "projects" / project / "models" / n / "labels.csv") for n in cls)
    vers = sorted({_det_version(n) for n in det})
    sha = shas.most_common(1)[0][0] if shas else "none"
    dest = (f"{today or time.strftime('%Y-%m-%d')}_cls-{sha}_det-{'-'.join(vers) or 'none'}_"
            f"{re.sub(r'[^A-Za-z0-9-]+', '-', reason).strip('-') or 'archive'}")
    return {"dest": dest, "classifiers": cls, "detectors": det, "kept": kept, "active_classifier": active_cls,
            "active_detector": det_w or "models/stage2_yolo/stage2_best.pt", "label_shas": dict(shas)}


def run(root: Path, project: str, p: dict) -> Path:
    """Move what plan() listed and write ARCHIVE_INFO.json. Refuses if the destination already exists."""
    out = root / "models_archive" / p["dest"]
    if out.exists():
        raise SystemExit(f"{out} already exists: choose another --reason")
    moved = []
    for n in p["classifiers"]:
        src = root / "projects" / project / "models" / n
        dst = out / f"{project}_models" / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append({"kind": "classifier", "name": n, "from": str(src.relative_to(root)), "to": str(dst.relative_to(root))})
    for n in p["detectors"]:
        src = root / "models" / "stage2_yolo" / n
        dst = out / "stage2_yolo" / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append({"kind": "detector", "name": n, "from": str(src.relative_to(root)), "to": str(dst.relative_to(root))})
        cand = root / "models" / "stage2_yolo" / f"candidate_{n.replace('bench_', '')}.json"
        if cand.exists():
            shutil.move(str(cand), str(out / "stage2_yolo" / cand.name))
    out.mkdir(parents=True, exist_ok=True)
    (out / "ARCHIVE_INFO.json").write_text(json.dumps({
        "archived": time.strftime("%Y-%m-%d %H:%M:%S"), "project": project, "dest": p["dest"],
        "labels_csv_sha8": p["label_shas"], "active_at_archive": {"classifier": p["active_classifier"],
                                                                   "detector": p["active_detector"]},
        "kept_in_place": p["kept"], "moved": moved,
        "restore": "python -m tools.archive_models --restore <this folder> <name>"}, indent=1))
    return out


def restore(root: Path, project: str, archive: str, name: str) -> Path:
    """Copy one archived model back where the app looks for it. Never overwrites."""
    a = root / "models_archive" / archive
    for src, dst in ((a / f"{project}_models" / name, root / "projects" / project / "models" / name),
                     (a / "om_bottle_models" / name, root / "projects" / project / "models" / name),
                     (a / "stage2_yolo" / name, root / "models" / "stage2_yolo" / name)):
        if src.exists():
            if dst.exists():
                raise SystemExit(f"{dst} already exists: not overwritten")
            (shutil.copytree if src.is_dir() else shutil.copy2)(str(src), str(dst))
            return dst
    raise SystemExit(f"{name!r} not found in {a}")


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        m = r / "projects" / "p" / "models"
        for n, lab in (("20260101-000000", "a"), ("20260102-000000", "a"), ("20260103-000000", "b")):
            (m / n).mkdir(parents=True)
            (m / n / "model.pt").write_bytes(b"x")
            (m / n / "labels.csv").write_text(lab)
        (r / "projects" / "p" / "config.json").write_text(json.dumps({"active_model": "20260103-000000"}))
        y = r / "models" / "stage2_yolo"
        for n in ("bench_yolov8n_v3", "bench_yolov8s_v3", "run_yolov8n"):
            (y / n / "weights").mkdir(parents=True)
            (y / n / "weights" / "best.pt").write_bytes(b"w")
        (y / "candidate_yolov8n_v3.json").write_text("{}")
        (y / "stage2_best.pt").write_bytes(b"w")
        (r / "settings.json").write_text(json.dumps({"detector_weights": str(y / "bench_yolov8s_v3/weights/best.pt")}))
        p = plan(r, "p", reason="pre v4", today="2026-10-13")
        sha_a = hashlib.sha256(b"a").hexdigest()[:8]
        assert p["dest"] == f"2026-10-13_cls-{sha_a}_det-v1-v3_pre-v4", p["dest"]
        assert p["classifiers"] == ["20260101-000000", "20260102-000000"], p          # active one is kept
        assert p["detectors"] == ["bench_yolov8n_v3", "run_yolov8n"], p               # active detector is kept
        assert "20260103-000000" in p["kept"] and "bench_yolov8s_v3" in p["kept"]
        out = run(r, "p", p)
        assert (m / "20260103-000000" / "model.pt").exists() and not (m / "20260101-000000").exists()
        assert (y / "bench_yolov8s_v3").exists() and (y / "stage2_best.pt").exists()
        info = json.loads((out / "ARCHIVE_INFO.json").read_text())
        assert len(info["moved"]) == 4 and (out / "stage2_yolo" / "candidate_yolov8n_v3.json").exists()
        back = restore(r, "p", p["dest"], "20260101-000000")
        assert back.exists() and (out / "p_models" / "20260101-000000").exists()         # copied, archive intact
        try:
            run(r, "p", p)
        except SystemExit:
            pass
        else:
            raise AssertionError("re-used an existing archive folder")
    print("ok  archive_models: named by training data, active classifier + detector never moved, "
          "ARCHIVE_INFO.json, restore copies back without overwriting")


def main(argv) -> None:
    if "--selftest" in argv:
        _selftest()
        return
    project = (ROOT / "projects" / "active.txt").read_text().strip()
    if "--restore" in argv:
        i = argv.index("--restore")
        print("restored ->", restore(ROOT, project, argv[i + 1], argv[i + 2]))
        return
    reason = argv[argv.index("--reason") + 1] if "--reason" in argv else "archive"
    p = plan(ROOT, project, reason=reason)
    print(f"destination  models_archive/{p['dest']}/")
    print(f"active (kept) classifier {p['active_classifier']}   detector {p['active_detector']}")
    print(f"classifiers to move ({len(p['classifiers'])}): {', '.join(p['classifiers']) or '-'}")
    print(f"detector runs to move ({len(p['detectors'])}): {', '.join(p['detectors']) or '-'}")
    print(f"kept in place: {', '.join(p['kept']) or '-'}")
    if "--run" in argv:
        print("archived ->", run(ROOT, project, p))
    else:
        print("(plan only: nothing moved; add --run)")


if __name__ == "__main__":
    main(sys.argv)
