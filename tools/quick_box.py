"""Quick box tool: one big picture, three classes, no panels. Writes the same annotations.json the Annotation Studio reads.

    python -m tools.quick_box --project enclosure_seg
    python -m tools.quick_box --selftest

Draw with the left mouse button (drag corner to corner). The class advances by itself: bottle -> cap -> label -> bottle ...
    1 / 2 / 3   pick bottle / cap / label        right-click   delete the box under the cursor
    Backspace   delete the last box              C             copy the boxes of the previous image (bursts barely move)
    Space / D   save + next image                A             previous image
    X           mark this image BAD (hand, blur): saved as skipped, no boxes
    G           go to the next image with no boxes yet           Q / Esc   save and quit
No cap on a missing-cap bottle and no label on a missing-label bottle: just do not draw them.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vision import annotate as A                                       # noqa: E402

CLASSES = ("bottle", "cap", "label")
COL = {"bottle": (60, 60, 255), "cap": (0, 220, 255), "label": (255, 0, 255)}
WIN_W, WIN_H = 1600, 900
MIN_PX = 6


def to_norm(x0, y0, x1, y1, scale, ox, oy, w, h):
    """Window drag (two corners) -> YOLO-style normalized box dict parts, or None when too small."""
    ax, bx = sorted((x0, x1))
    ay, by = sorted((y0, y1))
    if bx - ax < MIN_PX or by - ay < MIN_PX:
        return None
    nx0, nx1 = (ax - ox) / scale / w, (bx - ox) / scale / w
    ny0, ny1 = (ay - oy) / scale / h, (by - oy) / scale / h
    nx0, nx1, ny0, ny1 = (min(max(v, 0.0), 1.0) for v in (nx0, nx1, ny0, ny1))
    if nx1 - nx0 <= 0 or ny1 - ny0 <= 0:
        return None
    return {"x": round((nx0 + nx1) / 2, 6), "y": round((ny0 + ny1) / 2, 6),
            "w": round(nx1 - nx0, 6), "h": round(ny1 - ny0, 6)}


def box_to_win(b, scale, ox, oy, w, h):
    return (round(ox + (b["x"] - b["w"] / 2) * w * scale), round(oy + (b["y"] - b["h"] / 2) * h * scale),
            round(ox + (b["x"] + b["w"] / 2) * w * scale), round(oy + (b["y"] + b["h"] / 2) * h * scale))


def _num(p) -> int:
    m = re.search(r"(\d+)", Path(p).stem)
    return int(m.group(1)) if m else 0


class Tool:
    def __init__(self, slug: str):
        self.dir = ROOT / "projects" / slug
        self.path = self.dir / "annotations.json"
        self.data = A.load_annotations(self.path)
        for c in CLASSES:
            if c not in self.data["classes"]:
                raise SystemExit(f"class {c!r} missing in {self.path}")
        self.rels = sorted((f"_inbox/{p.name}" for p in (self.dir / "images" / "_inbox").glob("*.jpg")),
                           key=_num)
        self.i = self.first_open()
        self.cls = 0
        self.boxes: list = []
        self.drag = None
        self.cur = None
        self.copy_src = None
        self.im = None
        self.scale, self.ox, self.oy = 1.0, 0, 0
        self.load()

    # ---- data
    def entry(self, rel):
        return self.data["images"].get(rel)

    def done(self, rel) -> bool:
        e = self.entry(rel)
        return bool(e and e.get("reviewed"))

    def first_open(self, start: int = 0) -> int:
        for k in range(len(self.rels)):
            j = (start + k) % len(self.rels)
            if not self.done(self.rels[j]):
                return j
        return 0

    def load(self):
        rel = self.rels[self.i]
        full = cv2.imread(str(self.dir / "images" / rel), cv2.IMREAD_REDUCED_COLOR_2)
        self.h, self.w = full.shape[:2]
        self.scale = min((WIN_W - 20) / self.w, (WIN_H - 90) / self.h)
        self.im = cv2.resize(full, (int(self.w * self.scale), int(self.h * self.scale)), interpolation=cv2.INTER_AREA)
        self.ox, self.oy = (WIN_W - self.im.shape[1]) // 2, 40
        e = self.entry(rel) or {}
        self.boxes = [dict(b) for b in e.get("boxes") or []]
        self.bad = bool(e.get("bad"))
        self.cls = 0

    def save(self):
        rel = self.rels[self.i]
        e = {"reviewed": True, "boxes": [dict(b) for b in self.boxes], "polygons": []}
        if self.bad:
            e["bad"] = True
        self.data["images"][rel] = e
        A.save_annotations(self.path, self.data)

    def go(self, j):
        self.save()
        self.i = max(0, min(len(self.rels) - 1, j))
        self.load()

    # ---- interaction
    def add(self, nb):
        nb["cls"] = CLASSES[self.cls]
        self.boxes.append(nb)
        self.bad = False
        self.cls = (self.cls + 1) % len(CLASSES)

    def mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drag, self.cur = (x, y), (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            self.cur = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            nb = to_norm(*self.drag, x, y, self.scale, self.ox, self.oy, self.w, self.h)
            self.drag = self.cur = None
            if nb:
                self.add(nb)
        elif ev == cv2.EVENT_RBUTTONDOWN:
            for k in range(len(self.boxes) - 1, -1, -1):
                x0, y0, x1, y1 = box_to_win(self.boxes[k], self.scale, self.ox, self.oy, self.w, self.h)
                if x0 <= x <= x1 and y0 <= y <= y1:
                    del self.boxes[k]
                    break

    def draw(self):
        canvas = np.full((WIN_H, WIN_W, 3), 28, np.uint8)
        canvas[self.oy:self.oy + self.im.shape[0], self.ox:self.ox + self.im.shape[1]] = self.im
        for b in self.boxes:
            x0, y0, x1, y1 = box_to_win(b, self.scale, self.ox, self.oy, self.w, self.h)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), COL.get(b["cls"], (0, 255, 0)), 2)
            cv2.putText(canvas, b["cls"], (x0 + 3, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL.get(b["cls"], (0, 255, 0)), 2)
        if self.drag and self.cur:
            cv2.rectangle(canvas, self.drag, self.cur, COL[CLASSES[self.cls]], 1)
        n_done = sum(1 for r in self.rels if self.done(r))
        rel = self.rels[self.i]
        cv2.putText(canvas, f"{self.i + 1}/{len(self.rels)}  {Path(rel).name}   done {n_done}   "
                            f"{'BAD (skipped)' if self.bad else ''}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        nxt = CLASSES[self.cls]
        cv2.putText(canvas, f"NEXT: {nxt}", (WIN_W - 250, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL[nxt], 2)
        cv2.putText(canvas, "drag = box | 1 2 3 class | Space next | A back | C copy prev | Backspace undo | RClick delete | X bad | Q quit",
                    (12, WIN_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return canvas

    def run(self):
        cv2.namedWindow("quick box", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("quick box", self.mouse)
        while True:
            cv2.imshow("quick box", self.draw())
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                self.save()
                break
            if k in (ord(" "), ord("d")):
                self.go(self.i + 1)
            elif k == ord("a"):
                self.go(self.i - 1)
            elif k in (ord("1"), ord("2"), ord("3")):
                self.cls = k - ord("1")
            elif k == 8 and self.boxes:
                self.boxes.pop()
                self.cls = max(0, self.cls - 1)
            elif k == ord("c") and self.i > 0:
                prev = self.entry(self.rels[self.i - 1]) or {}
                self.boxes = [dict(b) for b in prev.get("boxes") or []]
                self.bad = False
            elif k == ord("x"):
                self.boxes, self.bad = [], True
                self.go(self.i + 1)
            elif k == ord("g"):
                self.save()
                self.i = self.first_open(self.i + 1)
                self.load()
        cv2.destroyAllWindows()


def selftest():
    nb = to_norm(110, 140, 310, 540, 0.5, 10, 40, 1920, 1080)
    assert nb and abs(nb["x"] - ((110 - 10) / 0.5 / 1920 + (310 - 10) / 0.5 / 1920) / 2) < 1e-6
    x0, y0, x1, y1 = box_to_win(nb, 0.5, 10, 40, 1920, 1080)
    assert (x0, y0, x1, y1) == (110, 140, 310, 540)
    assert to_norm(5, 5, 8, 8, 1, 0, 0, 100, 100) is None
    print("quick_box self-test ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="enclosure_seg")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        Tool(a.project).run()
