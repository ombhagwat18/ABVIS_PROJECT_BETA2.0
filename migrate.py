"""One-shot: move the single-project layout into projects/<slug>/.

    All Datasets/Good Bottle/  ->  projects/om-bottle/images/+ve/
    All Datasets/<Other>/      ->  projects/om-bottle/images/-ve/<Other>/
    All Datasets/_captured/    ->  projects/om-bottle/images/_inbox/
    data/*, models/*           ->  projects/om-bottle/

Every existing label and reviewed flag is kept: labels.csv is rewritten with the
new paths rather than re-imported. A re-import would reset each image to the one
defect its folder implies and throw away every second defect ticked by hand.

Rename, not copy -- this is ~1 GB, and a rename on the same volume is instant.

    python migrate.py            # show what would move
    python migrate.py --run
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import dataset as D

ROOT = Path(__file__).resolve().parent
OLD_IMAGES = ROOT / "All Datasets"
OLD_DATA = ROOT / "data"
OLD_MODELS = ROOT / "models"

# The folder that means "no defect", and the one that means "not looked at yet".
GOOD_FOLDER = "Good Bottle"
CAPTURED_FOLDER = "_captured"


def plan(title="OM Bottle") -> tuple[str, dict[str, str]]:
    """-> (slug, {old top-level folder: new path relative to images/})."""
    name = D.slug(title)
    moves = {}
    if OLD_IMAGES.is_dir():
        for p in sorted(OLD_IMAGES.iterdir()):
            if not p.is_dir():
                continue
            if p.name == GOOD_FOLDER:
                moves[p.name] = D.POS_DIR
            elif p.name == CAPTURED_FOLDER:
                moves[p.name] = D.INBOX_DIR
            else:
                moves[p.name] = f"{D.NEG_DIR}/{p.name}"
    return name, moves


def remap(old_rel: str, moves: dict[str, str]) -> str | None:
    """'Tilt Cap/snap_1.jpg' -> '-ve/Tilt Cap/snap_1.jpg'."""
    top, _, rest = old_rel.partition("/")
    if top not in moves or not rest:
        return None
    return f"{moves[top]}/{rest}"


def run(title="OM Bottle", dry=True) -> str:
    name, moves = plan(title)
    dest = D.PROJECTS / name
    if not moves:
        raise SystemExit(f"nothing to migrate: {OLD_IMAGES} has no folders")
    if dest.exists():
        raise SystemExit(f"{dest} already exists -- migrate has already run")

    print(f"project  {title!r}  ->  {dest}")
    for old, new in moves.items():
        n = sum(1 for _ in (OLD_IMAGES / old).rglob("*") if _.is_file())
        print(f"  {old:<22} -> images/{new:<28} {n:>5} files")
    for extra, where in ((OLD_DATA, "."), (OLD_MODELS, "models")):
        if extra.is_dir():
            print(f"  {extra.name + '/':<22} -> {where}")
    if dry:
        print("\ndry run. re-run with --run to move the files.")
        return name

    (dest / "images").mkdir(parents=True)
    (dest / "project.json").write_text(json.dumps({"title": title}, indent=2), encoding="utf-8")
    for old, new in moves.items():
        target = dest / "images" / new
        target.parent.mkdir(parents=True, exist_ok=True)
        (OLD_IMAGES / old).rename(target)
    for sub in (D.POS_DIR, D.NEG_DIR, D.INBOX_DIR):
        (dest / "images" / sub).mkdir(parents=True, exist_ok=True)

    if OLD_MODELS.is_dir():
        OLD_MODELS.rename(dest / "models")
    (dest / "models").mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    if (OLD_DATA / "labels.csv").exists():
        with (OLD_DATA / "labels.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        out = [rows[0]]
        for r in rows[1:]:
            if not r:
                continue
            new = remap(r[0], moves)
            if new is None:
                dropped += 1                 # a loose file, not under any folder
                continue
            out.append([new, *r[1:]])
            kept += 1
        with (dest / "labels.csv").open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(out)

    for keep in ("config.json", "scenes.json", "rejects.csv"):
        if (OLD_DATA / keep).exists():
            shutil.move(str(OLD_DATA / keep), str(dest / keep))
    for cache in ("cache", "thumbs"):
        if (OLD_DATA / cache).is_dir():
            (OLD_DATA / cache).rename(dest / cache)

    D.use_project(name)
    empty = D.PROJECTS / "default"
    if empty.is_dir() and not any((empty / "images").rglob("*.jpg")):
        shutil.rmtree(empty, ignore_errors=True)   # the placeholder _boot() made

    defects, labels = D.load_labels()
    print(f"\nmoved. {kept} labelled rows kept"
          + (f", {dropped} loose rows dropped" if dropped else "")
          + f"\n{len(labels)} images, {len(defects)} classes: {', '.join(defects)}")
    print(f"active project is now {name!r}. 'All Datasets/' and 'data/' are empty; delete them.")
    return name


def demo():
    """Self-check: labels must survive the move, on a fake tree in a temp dir."""
    global OLD_IMAGES, OLD_DATA, OLD_MODELS
    import tempfile
    was = (OLD_IMAGES, OLD_DATA, OLD_MODELS, D.PROJECTS, D.ACTIVE_TXT, D.PROJECT)
    try:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            OLD_IMAGES, OLD_DATA, OLD_MODELS = t / "All Datasets", t / "data", t / "models"
            D.PROJECTS, D.ACTIVE_TXT = t / "projects", t / "projects" / "active.txt"
            for folder, fn in ((GOOD_FOLDER, "g1.jpg"), ("Tilt Cap", "t1.jpg"),
                               (CAPTURED_FOLDER, "c1.jpg")):
                (OLD_IMAGES / folder).mkdir(parents=True)
                (OLD_IMAGES / folder / fn).write_bytes(b"x")
            OLD_DATA.mkdir()
            (OLD_DATA / "labels.csv").write_text(
                "path,tilt_cap,water_level,reviewed\n"
                f"{GOOD_FOLDER}/g1.jpg,0,0,1\n"
                "Tilt Cap/t1.jpg,1,1,1\n"                 # the hand-added 2nd defect
                f"{CAPTURED_FOLDER}/c1.jpg,0,0,0\n", encoding="utf-8")

            run("OM Bottle", dry=False)
            rows = {r[0]: r[1:] for r in
                    csv.reader((D.PROJECTS / "om_bottle" / "labels.csv").open(newline=""))}
            assert f"{D.POS_DIR}/g1.jpg" in rows, rows
            assert rows[f"{D.NEG_DIR}/Tilt Cap/t1.jpg"] == ["1", "1", "1"], \
                "migration lost the second defect"
            assert rows[f"{D.INBOX_DIR}/c1.jpg"] == ["0", "0", "0"]
            assert not any(OLD_IMAGES.iterdir()), "source folders were copied, not moved"
    finally:
        OLD_IMAGES, OLD_DATA, OLD_MODELS = was[0], was[1], was[2]
        D.PROJECTS, D.ACTIVE_TXT = was[3], was[4]
        if was[5]:
            D.use_project(was[5])
    print("ok  migration keeps every label and moves rather than copies")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        title = next((a for a in sys.argv[1:] if not a.startswith("-")), "OM Bottle")
        run(title, dry="--run" not in sys.argv)
