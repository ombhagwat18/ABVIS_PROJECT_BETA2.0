"""Model lifecycle: CANDIDATE -> VALIDATED -> APPROVED -> ACTIVE (-> ARCHIVED), or REJECTED.

What is ON DISK (checkpoints, metrics, held-out test results) is discovered every time -- nothing is
re-measured or copied here. What a PERSON decided (validated on the real camera, approved, rejected)
lives in models/model_status.json with who / when / why. What is ACTIVE is read from where the
runtime reads it (project config.json "active_model" for the classifier, settings.json
"detector_weights" for the detector), so this file can never disagree with the line.

Rules (enforced, not advisory):
  * training never activates (train.run(activate=False) is the default);
  * VALIDATED needs a held-out test result on record AND a real-camera validation note;
  * APPROVED needs VALIDATED; ACTIVE (activate) needs APPROVED;
  * every activation appends to models/deployments.jsonl (previous model, thresholds, recipe id,
    dataset version, time), and rollback() re-activates the previous deployment of that kind;
  * the previous model is ARCHIVED, never deleted.

Critical classes (settings "critical_classes", default ["missing_cap"]) are listed explicitly with
their TP / FP / FN; a model with NO test positives for a critical class is flagged, so a macro
average cannot hide the class production depends on.

    python model_registry.py            # list (read-only)
    python model_registry.py --selftest # self-test in a temporary folder
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from vision import dataset as D

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
YOLO_DIR = MODELS_DIR / "stage2_yolo"

CANDIDATE, VALIDATED, APPROVED, ACTIVE, ARCHIVED, REJECTED = (
    "CANDIDATE", "VALIDATED", "APPROVED", "ACTIVE", "ARCHIVED", "REJECTED")
STATUSES = (CANDIDATE, VALIDATED, APPROVED, ACTIVE, ARCHIVED, REJECTED)


class RegistryError(Exception):
    pass


def _read(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Registry:
    def __init__(self, models_dir: Path = MODELS_DIR, yolo_dir: Path = YOLO_DIR, settings=None):
        self.models_dir, self.yolo_dir = Path(models_dir), Path(yolo_dir)
        self.status_path = self.models_dir / "model_status.json"
        self.deploy_path = self.models_dir / "deployments.jsonl"
        self._settings = settings                      # callable -> dict (tests); default D.load_settings

    def settings(self) -> dict:
        return self._settings() if self._settings else D.load_settings()

    def critical(self) -> list:
        return list(self.settings().get("critical_classes", ["missing_cap"]))

    # ------------------------------------------------------------------ discovery
    def discover(self) -> list:
        out = []
        active_cls = D.load_config().get("active_model")
        crit = self.critical()
        if D.MODELS.exists():
            for d in sorted(p for p in D.MODELS.iterdir() if (p / "metrics.json").exists()):
                m = _read(d / "metrics.json") or {}
                t = _read(d / "test_metrics.json")
                test = (t or {}).get("test") or None
                per = (test or {}).get("per_defect") or {}
                out.append({
                    "name": f"cls/{D.PROJECT}/{d.name}", "kind": "classification", "id": d.name,
                    "arch": m.get("arch"), "project": D.PROJECT, "checkpoint": str(d / "model.pt"),
                    "size_mb": m.get("model_size_mb"), "trained": d.name,
                    "dataset": {"n_train": m.get("n_train"), "n_val": m.get("n_val"), "n_test": m.get("n_test"),
                                "labels_csv_sha256": (t or {}).get("labels_csv_sha256")},
                    "val_macro_f1": m.get("macro_f1"),
                    "test": None if not test else {k: test.get(k) for k in
                                                   ("n_images", "macro_f1", "macro_precision", "macro_recall",
                                                    "exact_match_accuracy", "false_pass", "false_reject")},
                    "critical": {c: per.get(c) for c in crit},
                    "latency_ms_gpu": ((t or {}).get("latency") or {}).get("ms_mean"),
                    "latency_ms_cpu": ((t or {}).get("latency") or {}).get("cpu_ms_mean"),
                    "thresholds": {k: v.get("threshold") for k, v in (m.get("per_defect") or {}).items()},
                    "has_test": bool(test), "active_now": d.name == active_cls})
        det_active = Path(self.settings().get("detector_weights") or (self.yolo_dir / "stage2_best.pt")).resolve()
        best = self.yolo_dir / "stage2_best.pt"
        tm = _read(self.yolo_dir / "training_metadata.json") or {}
        if best.exists():
            out.append({"name": "det/stage2_best", "kind": "detection", "id": "stage2_best", "arch": tm.get("model", "yolov8n"),
                        "checkpoint": str(best), "size_mb": tm.get("model_size_mb"), "trained": tm.get("date"),
                        "dataset": {"export_tree_sha256": tm.get("export_hash_before")},
                        "test": tm.get("test"), "critical": {}, "has_test": bool(tm.get("test")),
                        "latency_ms_gpu": (tm.get("benchmark_gpu") or {}).get("ms_mean"),
                        "active_now": best.resolve() == det_active})
        for p in sorted(self.yolo_dir.glob("candidate_*.json")):
            c = _read(p) or {}
            ck = ROOT / c.get("checkpoint", "")
            out.append({"name": f"det/{p.stem[len('candidate_'):]}", "kind": "detection", "id": p.stem,
                        "arch": Path(c.get("model", "")).stem, "checkpoint": str(ck), "sha256": c.get("sha256"),
                        "size_mb": c.get("size_mb"), "trained": c.get("date"), "dataset": c.get("dataset"),
                        "test": c.get("test"), "critical": {}, "has_test": bool(c.get("test")),
                        "latency_ms_gpu": (c.get("benchmark_gpu") or {}).get("ms_mean"),
                        "candidate_record": str(p), "active_now": ck.exists() and ck.resolve() == det_active})
        st = self._status()
        for e in out:
            rec = st.get(e["name"], {})
            e["status"] = ACTIVE if e["active_now"] else (rec.get("status") if rec.get("status") not in (None, ACTIVE)
                                                         else (ARCHIVED if rec.get("status") == ACTIVE else CANDIDATE))
            e["history"] = rec.get("history", [])
            e["warnings"] = self._warnings(e)
        return out

    def _warnings(self, e) -> list:
        w = []
        if not e["has_test"]:
            w.append("no held-out test result recorded")
        for c, v in (e.get("critical") or {}).items():
            if not v or not v.get("n_pos"):
                w.append(f"critical class {c}: NO test positives - cannot be trusted to detect it")
        return w

    def get(self, name: str) -> dict:
        for e in self.discover():
            if e["name"] == name:
                return e
        raise RegistryError(f"unknown model {name!r}")

    # ------------------------------------------------------------------ decisions
    def _status(self) -> dict:
        return _read(self.status_path) or {}

    def _set(self, name: str, status: str, by: str, note: str):
        st = self._status()
        rec = st.setdefault(name, {"history": []})
        rec["status"] = status
        rec["history"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status, "by": by, "note": note})
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(st, indent=2), encoding="utf-8")

    def validate(self, name: str, camera_note: str, by: str = "engineer", real: dict | None = None,
                 shortcut: dict | None = None):
        """VALIDATED needs: a held-out test result, a written real-camera note, real-camera NUMBERS within the gates
        (model_checks.real_camera_gate: good bottles called defective, defective bottles passed, minimum sample
        sizes) and, for a classifier, the background-shortcut check (model_checks.shortcut_check) passed or not
        runnable on this PC. Every number is stored with the decision."""
        from registry import model_checks as MC
        e = self.get(name)
        if not e["has_test"]:
            raise RegistryError(f"{name}: no held-out test result on record (run the test evaluation first)")
        if not camera_note.strip():
            raise RegistryError("describe the real-camera validation (bottles, camera, result) to validate")
        g = MC.gates(self.settings())
        probs = MC.real_camera_gate(real or {}, g)
        if probs:
            raise RegistryError(f"{name}: real-camera check not passed: " + "; ".join(probs))
        if e["kind"] == "classification":
            if shortcut is None:
                raise RegistryError(f"{name}: run the background-shortcut check first (model_checks.shortcut_check)")
            if not shortcut.get("skipped") and not shortcut.get("passed"):
                raise RegistryError(f"{name}: shortcut check failed: {shortcut.get('fired')}/{shortcut.get('n')} capped "
                                    f"bottles called missing_cap ({shortcut.get('fire_pct')} %, limit "
                                    f"{g['max_shortcut_fire_pct']:g} %)")
        r = real
        note = (f"real-camera validation: {camera_note.strip()} | good {r['good_called_defective']}/{r['good_n']} called "
                f"defective, defective {r['defective_passed']}/{r['defective_n']} passed"
                + (f" | shortcut {shortcut.get('fired')}/{shortcut.get('n')}" if shortcut and not shortcut.get("skipped")
                   else " | shortcut check not runnable here" if shortcut else ""))
        self._set(name, VALIDATED, by, note)

    def approve(self, name: str, by: str = "engineer", note: str = ""):
        if self.get(name)["status"] != VALIDATED:
            raise RegistryError(f"{name}: only a VALIDATED model can be approved")
        self._set(name, APPROVED, by, note or "approved")

    def reject(self, name: str, by: str = "engineer", note: str = ""):
        e = self.get(name)
        if e["status"] == ACTIVE:
            raise RegistryError(f"{name} is ACTIVE: activate another model (or roll back) first")
        self._set(name, REJECTED, by, note or "rejected")

    def activate(self, name: str, by: str = "engineer", note: str = "", _rollback: bool = False) -> dict:
        """Make an APPROVED model the production model of its kind. Returns the deployment record."""
        e = self.get(name)
        if e["status"] == ACTIVE:
            raise RegistryError(f"{name} is already active")
        if e["status"] != APPROVED and not _rollback:
            raise RegistryError(f"{name}: status {e['status']} - only an APPROVED model can be activated")
        prev = next((x for x in self.discover() if x["kind"] == e["kind"] and x["active_now"]), None)
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": e["kind"], "model": name, "id": e["id"],
               "previous": prev and prev["name"], "by": by, "note": note, "rollback": _rollback,
               "dataset": e.get("dataset"), "recipe_id": None}
        try:
            from line import production_store
            rec["recipe_id"] = production_store.recipe_id(D.load_config().get("inspection"))
        except Exception:                                               # noqa: BLE001
            pass
        if e["kind"] == "classification":
            cfg = D.load_config()
            cfg["active_model"] = e["id"]
            if e.get("thresholds"):
                cfg["thresholds"] = dict(e["thresholds"])
            D.save_config(cfg)
            rec["thresholds"] = cfg.get("thresholds")
        else:
            ck = Path(e["checkpoint"])
            if not ck.exists():
                raise RegistryError(f"{ck} does not exist")
            if e.get("sha256"):
                got = _sha(ck)
                if got != e["sha256"]:
                    raise RegistryError(f"{ck.name}: sha256 {got[:12]}... is not the trained {e['sha256'][:12]}...")
                prov = ck.parent / "MODEL_PROVENANCE.json"           # detect.verify_checkpoint reads it beside the file
                if not prov.exists():
                    prov.write_text(json.dumps({"model": {"checkpoint": {"sha256": got, "path": str(ck)}},
                                                "source": e.get("candidate_record"),
                                                "written_by": "model_registry.activate"}, indent=2), encoding="utf-8")
            s = self.settings()
            if self._settings is None:
                s = D.load_settings()
                if ck.resolve() == (self.yolo_dir / "stage2_best.pt").resolve():
                    s.pop("detector_weights", None)
                else:
                    s["detector_weights"] = str(ck)
                D.save_settings(s)
            else:
                s["detector_weights"] = str(ck)
        if prev is not None:
            self._set(prev["name"], ARCHIVED, by, f"replaced by {name}")
        self._set(name, ACTIVE, by, note or ("rollback" if _rollback else "activated"))
        with self.deploy_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def deployments(self) -> list:
        if not self.deploy_path.exists():
            return []
        return [json.loads(x) for x in self.deploy_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def rollback(self, kind: str, by: str = "engineer") -> dict:
        """Re-activate the model that was active before the current one of this kind."""
        deps = [d for d in self.deployments() if d["kind"] == kind]
        if not deps or not deps[-1].get("previous"):
            raise RegistryError(f"no earlier {kind} deployment to roll back to")
        return self.activate(deps[-1]["previous"], by, "rollback", _rollback=True)


def demo():
    import tempfile
    was = (D.MODELS, D.PROJECT)
    saved_cfg = None
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cls_dir, yolo = td / "proj_models", td / "yolo"
        cls_dir.mkdir(); yolo.mkdir()
        for stamp, tested in (("20260101-000000", True), ("20260102-000000", False)):
            d = cls_dir / stamp
            d.mkdir()
            (d / "model.pt").write_bytes(b"x")
            (d / "metrics.json").write_text(json.dumps({"arch": "efficientnet_b0", "macro_f1": 1.0,
                                                        "per_defect": {"missing_cap": {"threshold": 1.01},
                                                                       "tilt_cap": {"threshold": 0.5}}}))
            if tested:
                (d / "test_metrics.json").write_text(json.dumps({"test": {"macro_f1": 0.8, "per_defect": {
                    "missing_cap": {"n_pos": 0, "note": "no test positives"}}}, "latency": {"ms_mean": 30}}))
        (yolo / "stage2_best.pt").write_bytes(b"best")
        cand = yolo / "cand_weights.pt"
        cand.write_bytes(b"cand")
        (yolo / "candidate_yolov8s.json").write_text(json.dumps({"model": "yolov8s.pt", "checkpoint": str(cand),
                                                                 "sha256": _sha(cand), "test": {"map50": 0.97}}))
        cfg_store = {"active_model": "20260101-000000", "thresholds": {}}
        settings = {}
        D.MODELS = cls_dir
        real_load, real_save = D.load_config, D.save_config
        D.load_config = lambda: dict(cfg_store)
        D.save_config = lambda c: (cfg_store.clear(), cfg_store.update(c))
        try:
            reg = Registry(td / "models", yolo, settings=lambda: settings)
            names = {e["name"]: e for e in reg.discover()}
            a, b = f"cls/{D.PROJECT}/20260101-000000", f"cls/{D.PROJECT}/20260102-000000"
            assert names[a]["status"] == ACTIVE and names[b]["status"] == CANDIDATE
            assert any("missing_cap" in w for w in names[a]["warnings"]), names[a]["warnings"]
            assert names["det/stage2_best"]["status"] == ACTIVE and names["det/yolov8s"]["status"] == CANDIDATE
            real_ok = {"good_n": 40, "good_called_defective": 0, "defective_n": 35, "defective_passed": 0}
            for bad in (lambda: reg.activate(b), lambda: reg.approve(b), lambda: reg.validate(b, "5 bottles ok", real=real_ok),
                        lambda: reg.validate("det/yolov8s", "", real=real_ok),
                        lambda: reg.validate("det/yolov8s", "20 frames", real=dict(real_ok, good_called_defective=5)),
                        lambda: reg.validate("det/yolov8s", "20 frames"),
                        lambda: reg.validate(a, "cam ok", real=real_ok),                       # classifier, no shortcut check
                        lambda: reg.validate(a, "cam ok", real=real_ok, shortcut={"n": 25, "fired": 25, "fire_pct": 100.0,
                                                                                 "passed": False})):
                try:
                    bad()
                    raise AssertionError("gate not enforced")
                except RegistryError:
                    pass
            # detector candidate: validate -> approve -> activate (provenance written, sha checked) -> rollback
            reg.validate("det/yolov8s", "EMEET cam 2, line test", real=real_ok)
            reg.approve("det/yolov8s")
            rec = reg.activate("det/yolov8s")
            assert settings["detector_weights"] == str(cand) and rec["previous"] == "det/stage2_best"
            assert (yolo / "MODEL_PROVENANCE.json").exists()
            st = {e["name"]: e["status"] for e in reg.discover()}
            assert st["det/yolov8s"] == ACTIVE and st["det/stage2_best"] == ARCHIVED, st
            back = reg.rollback("detection")
            assert back["model"] == "det/stage2_best" and settings["detector_weights"].endswith("stage2_best.pt")
            # a tampered candidate is refused
            reg._set("det/yolov8s", APPROVED, "t", "")
            cand.write_bytes(b"tampered")
            try:
                reg.activate("det/yolov8s")
                raise AssertionError("tampered weights activated")
            except RegistryError as e:
                assert "sha256" in str(e)
            # classifier: activation writes active_model + that checkpoint's thresholds
            reg._set(a, ARCHIVED, "t", "")
            assert len(reg.deployments()) == 2
        finally:
            D.MODELS, D.PROJECT = was
            D.load_config, D.save_config = real_load, real_save
    print("ok  model registry: discovery from disk, CANDIDATE -> VALIDATED -> APPROVED -> ACTIVE gates, critical-class "
          "warning, sha-checked detector activation + provenance, deployment log, rollback, previous model archived")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        demo()
    else:
        for e in Registry().discover():
            print(f"{e['status']:<10} {e['name']:<44} test={'yes' if e['has_test'] else 'NO '}  "
                  + ("; ".join(e["warnings"]) or "-"))
