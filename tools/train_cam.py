"""Train the classifier of one project from the terminal with a GPU-safe batch size.

    python -m tools.train_cam enclosure_cama [--epochs 25] [--batch 16]

vision.train.run() trains the ACTIVE project and leaves the result a CANDIDATE (nothing is activated).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import dataset as D                                           # noqa: E402
from vision import train as T                                             # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--arch", default=T.DEFAULT_ARCH)
    a = ap.parse_args()
    D.use_project(a.project)
    res = T.run(epochs=a.epochs, batch=a.batch, patience=a.patience, arch=a.arch, activate=False)
    print("RESULT", json.dumps({k: v for k, v in res.items() if isinstance(v, (int, float, str))}, indent=1, default=str))
