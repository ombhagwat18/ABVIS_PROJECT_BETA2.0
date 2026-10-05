"""Persistent production record: every bottle, every alarm, and the versions each run used.

    projects/<slug>/production/production.db     SQLite (stdlib, one file, survives restarts)
    projects/<slug>/production/evidence/<day>/   evidence images, by policy
    projects/<slug>/production/<day>.csv         the daily CSV machine_cycle already writes (kept)

Tables
  runs        one row per "Start inspection": job/product, recipe hash + body, model ids, the line
              settings in force -- so every bottle is traceable to the exact versions that judged it
  inspections one row per bottle (its final result), keyed by run_id + inspection_id, with the
              full machine_cycle.Bottle record as JSON, the PLC outcome and the evidence path
  alarms      one row per alarm change (raised / acknowledged / cleared)

Evidence policy (settings "evidence_policy"): NONE | ALL | REJECT_ONLY | FAULT_ONLY | REJECT_AND_FAULT
(default) | SAMPLE:<n> (every n-th bottle, plus every REJECT/FAULT). Images are written by one
background thread from a bounded queue, so a slow disk can never delay a PLC command; a dropped image
is recorded as dropped, never as written.

    python production_store.py     # self-test (temporary folder)
"""
from __future__ import annotations

import hashlib
import json
import queue
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import cv2

EVIDENCE_POLICIES = ("NONE", "ALL", "REJECT_ONLY", "FAULT_ONLY", "REJECT_AND_FAULT", "SAMPLE:10")
MIN_FREE_BYTES = 1 << 30                       # stop writing evidence below 1 GB free

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, started REAL, stopped REAL, project TEXT, job TEXT, product TEXT,
    recipe_id TEXT, recipe TEXT, models TEXT, settings TEXT, mode TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS inspections (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, inspection_id TEXT, wall REAL, day TEXT,
    final TEXT, decision TEXT, defects TEXT, confidence REAL, command TEXT, plc_status TEXT,
    status TEXT, evidence TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS ix_insp_day ON inspections(day, final);
CREATE INDEX IF NOT EXISTS ix_insp_run ON inspections(run_id, inspection_id);
CREATE TABLE IF NOT EXISTS alarms (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT, wall REAL, day TEXT, code TEXT, key TEXT, severity TEXT,
    state TEXT, message TEXT, cause TEXT, count INTEGER, run_id TEXT);
CREATE INDEX IF NOT EXISTS ix_alarm_day ON alarms(day);
"""


def recipe_id(recipe) -> str:
    """Short stable hash of a recipe (None = the built-in default recipe)."""
    if not recipe:
        return "default"
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]


DEFAULT_SHIFTS = (("A", "06:00", "14:00"), ("B", "14:00", "22:00"), ("C", "22:00", "06:00"))


def shift_span(day: str, shift) -> tuple:
    """(from, to) epoch seconds of shift (name, "HH:MM", "HH:MM") that STARTS on `day` (YYYY-MM-DD).
    A shift whose end is not after its start ends the next day (night shift)."""
    _, start, end = shift
    t0 = time.mktime(time.strptime(f"{day} {start}", "%Y-%m-%d %H:%M"))
    t1 = time.mktime(time.strptime(f"{day} {end}", "%Y-%m-%d %H:%M"))
    if t1 <= t0:
        t1 += 24 * 3600
    return t0, t1


def wants_evidence(policy: str, final: str, n: int) -> bool:
    p = (policy or "REJECT_AND_FAULT").upper()
    if p == "NONE":
        return False
    if p == "ALL":
        return True
    if p == "REJECT_ONLY":
        return final == "REJECT"
    if p == "FAULT_ONLY":
        return final == "FAULT"
    if p.startswith("SAMPLE:"):
        try:
            every = max(1, int(p.split(":", 1)[1]))
        except ValueError:
            every = 10
        return final in ("REJECT", "FAULT") or n % every == 0
    return final in ("REJECT", "FAULT")                      # REJECT_AND_FAULT (default)


class ProductionStore:
    def __init__(self, folder: Path, policy: str = "REJECT_AND_FAULT", on_problem=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "production.db"
        self.policy = policy
        self.on_problem = on_problem                         # fn(code, cause) -> alarms
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
        self._db.executescript(SCHEMA)
        self._db.commit()
        self.run_id: str | None = None
        self._n = 0
        self.dropped = 0
        self.written = 0
        self._q: "queue.Queue" = queue.Queue(maxsize=32)
        self._writer = threading.Thread(target=self._write_loop, name="evidence-writer", daemon=True)
        self._writer.start()

    # ------------------------------------------------------------------ runs
    def start_run(self, project: str, job: str, product: str, recipe, models: dict, settings: dict,
                  mode: str = "") -> str:
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (self.run_id, time.time(), None, project, job, product, recipe_id(recipe),
                              json.dumps(recipe), json.dumps(models), json.dumps(settings, default=str), mode, ""))
            self._db.commit()
        return self.run_id

    def stop_run(self):
        if self.run_id is None:
            return
        with self._lock:
            self._db.execute("UPDATE runs SET stopped=? WHERE run_id=?", (time.time(), self.run_id))
            self._db.commit()

    # ------------------------------------------------------------------ bottles
    def record(self, bottle) -> str:
        """Store one finished bottle (machine_cycle.Bottle). Returns the evidence path ('' if none)."""
        self._n += 1
        row = bottle.row()
        final = bottle.final or "FAULT"
        ev = ""
        img = getattr(bottle, "evidence", None)
        if img is None:
            img = getattr(bottle, "thumb", None)
        if img is not None and wants_evidence(self.policy, final, self._n):
            ev = self._queue_evidence(bottle, final, img, getattr(bottle, "thumb", None))
        day = time.strftime("%Y-%m-%d", time.localtime(bottle.wall))
        with self._lock:
            self._db.execute(
                "INSERT INTO inspections (run_id, inspection_id, wall, day, final, decision, defects, confidence, "
                "command, plc_status, status, evidence, data) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, bottle.inspection_id, bottle.wall, day, final, bottle.decision, row["defects"],
                 bottle.confidence, bottle.command, bottle.plc_status, bottle.status, ev,
                 json.dumps(row, default=str)))
            self._db.commit()
        return ev

    def _queue_evidence(self, bottle, final, img, overlay) -> str:
        day = time.strftime("%Y%m%d", time.localtime(bottle.wall))
        d = self.folder / "evidence" / day
        name = f"{self.run_id or 'run'}_{bottle.inspection_id}_{final}"
        try:
            if shutil.disk_usage(str(self.folder)).free < MIN_FREE_BYTES:
                self.dropped += 1
                self._problem("DISK_LOW", f"less than 1 GB free at {self.folder}: evidence not written")
                return ""
            self._q.put_nowait((d, name, img, overlay))
        except queue.Full:
            self.dropped += 1
            self._problem("LOG_WRITE_FAILED", "evidence writer busy: image dropped")
            return ""
        except OSError as e:
            self._problem("LOG_WRITE_FAILED", f"evidence: {e}")
            return ""
        return str((d / f"{name}.jpg").relative_to(self.folder))

    def _write_loop(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            d, name, img, overlay = item
            try:
                d.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(d / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if overlay is not None and overlay is not img:
                    cv2.imwrite(str(d / f"{name}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    self.written += 1
                else:
                    self._problem("LOG_WRITE_FAILED", f"evidence image {name} could not be encoded")
            except Exception as e:                                    # noqa: BLE001
                self._problem("LOG_WRITE_FAILED", f"evidence {name}: {type(e).__name__}: {e}")
            finally:
                self._q.task_done()

    def flush(self, timeout: float = 5.0):
        t_end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < t_end:
            time.sleep(0.01)

    # ------------------------------------------------------------------ alarms
    def log_alarm(self, a):
        t = a.cleared if a.state == "CLEARED" and a.cleared else (a.acked if a.state == "ACKNOWLEDGED" and a.acked
                                                                   else a.raised)
        with self._lock:
            self._db.execute("INSERT INTO alarms (wall, day, code, key, severity, state, message, cause, count, run_id) "
                             "VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (t, time.strftime("%Y-%m-%d", time.localtime(t)), a.code, a.key, a.severity, a.state,
                              a.message, a.cause, a.count, self.run_id))
            self._db.commit()

    # ------------------------------------------------------------------ queries (any thread)
    def recent(self, n: int = 200, day: str | None = None, final: str | None = None, span: tuple | None = None) -> list:
        q, args = "SELECT run_id, inspection_id, wall, final, decision, defects, confidence, command, plc_status, " \
                  "evidence, data FROM inspections", []
        where = []
        if day:
            where.append("day=?"); args.append(day)
        if final:
            where.append("final=?"); args.append(final)
        if span:
            where.append("wall >= ? AND wall < ?"); args += [float(span[0]), float(span[1])]
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY rowid DESC LIMIT ?"
        args.append(int(n))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        keys = ("run_id", "inspection_id", "wall", "final", "decision", "defects", "confidence", "command",
                "plc_status", "evidence", "data")
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            d["data"] = json.loads(d["data"]) if d["data"] else {}
            out.append(d)
        return out

    def summary_range(self, t_from: float, t_to: float, label: str = "") -> dict:
        """Same as summary(), for any wall-clock span (a shift that crosses midnight, a run, a week)."""
        with self._lock:
            by = dict(self._db.execute("SELECT final, COUNT(*) FROM inspections WHERE wall >= ? AND wall < ? "
                                       "GROUP BY final", (t_from, t_to)).fetchall())
            defects = self._db.execute("SELECT defects FROM inspections WHERE wall >= ? AND wall < ? AND defects != ''",
                                       (t_from, t_to)).fetchall()
            hours = self._db.execute("SELECT CAST(strftime('%H', wall, 'unixepoch', 'localtime') AS INTEGER), final, "
                                     "COUNT(*) FROM inspections WHERE wall >= ? AND wall < ? GROUP BY 1, 2",
                                     (t_from, t_to)).fetchall()
            nalarm = self._db.execute("SELECT COUNT(*) FROM alarms WHERE wall >= ? AND wall < ? AND state='ACTIVE'",
                                      (t_from, t_to)).fetchone()[0]
            ni = self._db.execute("SELECT COUNT(*) FROM inspections WHERE wall >= ? AND wall < ? AND status='NOT INSPECTED'",
                                  (t_from, t_to)).fetchone()[0]
        dist: dict = {}
        for (txt,) in defects:
            for d in txt.split(";"):
                if d:
                    dist[d] = dist.get(d, 0) + 1
        hourly: dict = {}
        for h, f, c in hours:
            hourly.setdefault(int(h), {})[f] = c
        total = sum(by.values())
        return {"day": label, "span": (t_from, t_to), "total": total, "PASS": by.get("PASS", 0),
                "REJECT": by.get("REJECT", 0), "FAULT": by.get("FAULT", 0), "not_inspected": ni,
                "defects": dict(sorted(dist.items(), key=lambda kv: -kv[1])), "hourly": hourly,
                "alarms_raised": nalarm}

    def summary(self, day: str) -> dict:
        with self._lock:
            by = dict(self._db.execute("SELECT final, COUNT(*) FROM inspections WHERE day=? GROUP BY final",
                                       (day,)).fetchall())
            defects = self._db.execute("SELECT defects FROM inspections WHERE day=? AND defects != ''",
                                       (day,)).fetchall()
            hours = self._db.execute("SELECT CAST(strftime('%H', wall, 'unixepoch', 'localtime') AS INTEGER), final, "
                                     "COUNT(*) FROM inspections WHERE day=? GROUP BY 1, 2", (day,)).fetchall()
            nalarm = self._db.execute("SELECT COUNT(*) FROM alarms WHERE day=? AND state='ACTIVE'",
                                      (day,)).fetchone()[0]
        dist: dict = {}
        for (txt,) in defects:
            for d in txt.split(";"):
                if d:
                    dist[d] = dist.get(d, 0) + 1
        hourly: dict = {}
        for h, f, c in hours:
            hourly.setdefault(int(h), {})[f] = c
        total = sum(by.values())
        return {"day": day, "total": total, "PASS": by.get("PASS", 0), "REJECT": by.get("REJECT", 0),
                "FAULT": by.get("FAULT", 0), "defects": dict(sorted(dist.items(), key=lambda kv: -kv[1])),
                "hourly": hourly, "alarms_raised": nalarm}

    def alarms(self, day: str | None = None, n: int = 300) -> list:
        q = "SELECT wall, code, key, severity, state, message, cause, count FROM alarms"
        args: list = []
        if day:
            q += " WHERE day=?"; args.append(day)
        q += " ORDER BY rowid DESC LIMIT ?"
        args.append(int(n))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [dict(zip(("wall", "code", "key", "severity", "state", "message", "cause", "count"), r)) for r in rows]

    def alarms_range(self, span=None, n: int = 1000) -> list:
        """Alarm changes in a wall-clock span (t_from, t_to), newest first."""
        q, args = "SELECT wall, code, key, severity, state, message, cause, count FROM alarms", []
        if span:
            q += " WHERE wall >= ? AND wall < ?"
            args += [float(span[0]), float(span[1])]
        q += " ORDER BY rowid DESC LIMIT ?"
        args.append(int(n))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [dict(zip(("wall", "code", "key", "severity", "state", "message", "cause", "count"), r)) for r in rows]

    def days(self, n: int = 60) -> list:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT DISTINCT day FROM inspections ORDER BY day DESC LIMIT ?",
                                                   (n,)).fetchall()]

    def runs(self, n: int = 50) -> list:
        with self._lock:
            rows = self._db.execute("SELECT run_id, started, stopped, job, product, recipe_id, models, mode FROM runs "
                                    "ORDER BY started DESC LIMIT ?", (n,)).fetchall()
        return [dict(zip(("run_id", "started", "stopped", "job", "product", "recipe_id", "models", "mode"), r))
                for r in rows]

    def close(self):
        self.flush(2.0)
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        with self._lock:
            self._db.close()

    def _problem(self, code, cause):
        if self.on_problem:
            try:
                self.on_problem(code, cause)
            except Exception:                                         # noqa: BLE001
                pass


def demo():
    import tempfile
    import numpy as np
    from machine_cycle import Bottle

    assert wants_evidence("REJECT_AND_FAULT", "REJECT", 1) and not wants_evidence("REJECT_AND_FAULT", "PASS", 1)
    assert wants_evidence("SAMPLE:5", "PASS", 10) and not wants_evidence("SAMPLE:5", "PASS", 11)
    assert wants_evidence("SAMPLE:5", "FAULT", 11) and not wants_evidence("NONE", "FAULT", 1)
    assert recipe_id(None) == "default" and len(recipe_id({"anchor": "bottle"})) == 12
    with tempfile.TemporaryDirectory() as td:
        problems = []
        st = ProductionStore(Path(td), on_problem=lambda c, m: problems.append(c))
        run = st.start_run("om_bottle", "JOB-1", "250 ml", None, {"detection": "yolov8n"}, {"plc_t0_s": 0.75})
        img = np.zeros((60, 40, 3), np.uint8)
        for i, (final, dfx) in enumerate((("PASS", []), ("REJECT", ["missing_cap"]), ("FAULT", [])), 1):
            b = Bottle(f"{i:06d}", i, 100.0 + i, time.time(), trigger_id=i)
            b.final = b.decision = final
            b.defects, b.command, b.plc_status = dfx, "PASS" if final == "PASS" else "REJECT", "ACKED"
            b.thumb = img
            ev = st.record(b)
            assert bool(ev) == (final != "PASS"), (final, ev)
        st.flush()
        day = time.strftime("%Y-%m-%d")
        s = st.summary(day)
        assert (s["total"], s["PASS"], s["REJECT"], s["FAULT"]) == (3, 1, 1, 1) and s["defects"] == {"missing_cap": 1}, s
        rows = st.recent(10)
        assert [r["inspection_id"] for r in rows] == ["000003", "000002", "000001"] and rows[0]["run_id"] == run
        assert rows[1]["data"]["defects"] == "missing_cap" and (Path(td) / rows[1]["evidence"]).exists()
        assert st.recent(10, final="REJECT")[0]["inspection_id"] == "000002"
        import alarms as AL
        m = AL.AlarmManager(on_change=st.log_alarm)
        m.raise_("BOTTLE_UNTRIGGERED", "X0 while M1 held", key="000004")
        m.acknowledge_all()
        al = st.alarms(day)
        assert [a["state"] for a in al] == ["CLEARED", "ACTIVE"] and al[1]["code"] == "BOTTLE_UNTRIGGERED", al
        st.stop_run()
        assert st.runs()[0]["run_id"] == run and json.loads(st.runs()[0]["models"]) == {"detection": "yolov8n"}
        assert st.days() == [day] and not problems
        now = time.time()
        sr = st.summary_range(now - 3600, now + 3600, "last hour")
        assert sr["total"] == 3 and sr["REJECT"] == 1 and st.summary_range(now + 10, now + 20)["total"] == 0
        assert len(st.recent(10, span=(now - 3600, now + 3600))) == 3 and not st.recent(10, span=(0, 1))
        a, b = shift_span("2026-10-05", ("C", "22:00", "06:00"))
        assert b - a == 8 * 3600 and time.strftime("%Y-%m-%d %H:%M", time.localtime(b)) == "2026-10-06 06:00"
        st.close()
        st2 = ProductionStore(Path(td))                 # survives a restart
        assert st2.summary(day)["total"] == 3
        st2.close()
    print("ok  production store: runs with versions, one row per bottle, evidence policy + async writer, alarms, "
          "daily summary, survives restart")


if __name__ == "__main__":
    demo()
