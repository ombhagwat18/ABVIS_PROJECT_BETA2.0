"""Read the production database in plain words, and export it: CSV (opens in Excel) and a printable HTML report.

The database is ONE SQLite file per product project:  projects/<project>/production/production.db
(SQLite is built into Python: no server, no install, a single file you can copy). It has three tables:

  runs         one row each time START INSPECTION is pressed: job, product, which models and which recipe judged the
               bottles, and whether the PLC was the SIMULATOR or the REAL one.
  inspections  one row per bottle, written when its final result is known: when, id, GOOD / DEFECT / FAULT, which
               defects, how sure, what was sent to the PLC and what the PLC answered, how long each step took,
               and the evidence picture (if the evidence policy kept one).
  alarms       one row per alarm change (raised, acknowledged, cleared): code, severity, cause, count.

Nothing is ever deleted by the application. The same facts are also in the daily CSV (production/<date>.csv).

This module is pure reading + writing files; it never changes the database.   python production_export.py  # self-test
"""
from __future__ import annotations

import csv
import html
import json
import time
from pathlib import Path

from vision import verdict as V

TABLES = ("inspections", "alarms", "runs")

EXPLAIN = {
    "inspections": [
        ("time", "When the bottle was sensed at the photo-eye."),
        ("bottle", "Inspection id: 000001, 000002 ... one per bottle, restarts at 1 for each run."),
        ("result", "GOOD, DEFECT or FAULT. FAULT = the system could not vouch for the bottle (it is rejected by default)."),
        ("defects", "What was wrong, in words (blank for a good bottle)."),
        ("confidence", "How sure the decision was, 0 - 1."),
        ("sent_to_plc", "What was told to the PLC: PASS (M0) or REJECT (M1)."),
        ("plc_answer", "What the PLC did: acknowledged, reject pulse seen, or the problem."),
        ("decide_ms", "Milliseconds from the photo-eye trigger to the decision."),
        ("evidence", "Picture file kept for this bottle (relative to the production folder), if any."),
        ("run", "Which START (see runs) this bottle belongs to."),
    ],
    "alarms": [
        ("time", "When the alarm was raised / acknowledged / cleared."),
        ("code", "Alarm code, e.g. CAMERA_STALE, REJECT_DEADLINE_MISSED."),
        ("severity", "CRITICAL, MAJOR, WARNING or INFO."),
        ("state", "ACTIVE, ACKNOWLEDGED or CLEARED."),
        ("message", "What it means for the operator."),
        ("cause", "The technical detail."),
        ("count", "How many times it happened while active."),
    ],
    "runs": [
        ("run", "Run id = date-time of START INSPECTION."),
        ("started / stopped", "When the line was started and stopped."),
        ("job / product", "What the operator typed on the Production page."),
        ("recipe", "Short fingerprint of the inspection recipe used (changes when the recipe changes)."),
        ("models", "Which classifier / detector judged the bottles."),
        ("plc", "SIMULATOR or REAL PLC."),
    ],
}

ABOUT = """HOW THE DATABASE WORKS

Where: projects/<project>/production/production.db  (one SQLite file per product; copy it to back it up).
Why SQLite: it is built into Python, needs no server and no installation, survives a power cut (writes are
transactions) and can be opened by any tool (DB Browser, Excel via CSV, Python).

When something is written:
  START INSPECTION -> one row in RUNS (job, product, models, recipe, SIMULATOR / REAL PLC).
  A bottle is finished -> one row in INSPECTIONS, and (by the evidence policy) a picture in production/evidence/.
  An alarm is raised, acknowledged or cleared -> one row in ALARMS.

What it is NOT: a live display (the Production page shows the live bottle), and it never changes by itself: rows are
only added. The old daily CSV files in production/ keep being written as well.

Export: pick the table and the time range, then 'Export CSV' (opens in Excel) or 'Production report' (a printable
page with totals, defect counts, bottles per hour, alarms and the latest bottles; open it and Print -> Save as PDF).
"""


def _fmt_time(t):
    return "" if t is None else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(t)))


def rows_for(store, table: str, span=None, result: str = "All", text: str = "", limit: int = 5000) -> tuple:
    """(columns, rows) for the Database page and the CSV export; every cell is plain text / number."""
    text = (text or "").strip().lower()
    if table == "inspections":
        cols = ["time", "bottle", "result", "defects", "confidence", "sent_to_plc", "plc_answer", "decide_ms",
                "evidence", "run"]
        out = []
        for r in store.recent(limit, span=span, final=None if result in ("All", "") else
                              {"GOOD": "PASS", "DEFECT": "REJECT"}.get(result, result)):
            d = r.get("data") or {}
            tms = None
            for kv in (d.get("timings") or "").split(";"):
                if kv.startswith("trigger_to_decision_ms="):
                    tms = kv.split("=", 1)[1]
            row = [_fmt_time(r["wall"]), r["inspection_id"], V.shown_result(r["final"]),
                   ", ".join(V.pretty(x) for x in (r["defects"] or "").split(";") if x),
                   "" if r["confidence"] is None else round(r["confidence"], 3), r["command"] or "",
                   r["plc_status"] or "", tms or "", r["evidence"] or "", r["run_id"] or ""]
            if not text or text in " ".join(str(c) for c in row).lower():
                out.append(row)
        return cols, out
    if table == "alarms":
        cols = ["time", "code", "severity", "state", "message", "cause", "count"]
        rows = store.alarms_range(span, limit) if hasattr(store, "alarms_range") else []
        out = [[_fmt_time(a["wall"]), a["code"], a["severity"], a["state"], a["message"], a["cause"], a["count"]]
               for a in rows]
        return cols, [r for r in out if not text or text in " ".join(str(c) for c in r).lower()]
    if table == "runs":
        cols = ["run", "started", "stopped", "job", "product", "recipe", "models", "plc"]
        out = []
        for r in store.runs(1000):
            if span and not (span[0] <= (r["started"] or 0) < span[1]):
                continue
            try:
                models = ", ".join(f"{k}: {v}" for k, v in json.loads(r["models"] or "{}").items() if v)
            except ValueError:
                models = r["models"] or ""
            out.append([r["run_id"], _fmt_time(r["started"]), _fmt_time(r["stopped"]), r["job"] or "", r["product"] or "",
                        r["recipe_id"] or "", models, r["mode"] or ""])
        return cols, [r for r in out if not text or text in " ".join(str(c) for c in r).lower()]
    raise ValueError(f"unknown table {table!r}")


def export_csv(store, table: str, path, span=None, result="All", text="") -> int:
    """Write the rows the Database page shows. utf-8 with BOM so Excel reads the characters right. Returns the row count."""
    cols, rows = rows_for(store, table, span, result, text, limit=1_000_000)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    return len(rows)


def report_html(store, span, title: str = "Production report", label: str = "") -> str:
    """A self-contained printable page (no external files)."""
    s = store.summary_range(span[0], span[1], label)
    total = s["total"]
    pct = lambda n: f"{100 * n / total:.1f} %" if total else "-"        # noqa: E731
    cols, rows = rows_for(store, "inspections", span, limit=200)
    _, alarms = rows_for(store, "alarms", span, limit=100)
    _, runs = rows_for(store, "runs", span)
    mx = max(s["defects"].values(), default=1)
    e = html.escape

    def table(headers, data, cls=""):
        th = "".join(f"<th>{e(h)}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>" for r in data)
        return f"<table class='{cls}'><tr>{th}</tr>{body or '<tr><td colspan=99>none</td></tr>'}</table>"
    bars = "".join(f"<tr><td>{e(V.pretty(d))}</td><td>{n}</td><td><div class='bar' style='width:{int(300 * n / mx)}px'>"
                   f"</div></td></tr>" for d, n in s["defects"].items()) or "<tr><td colspan=3>no defects</td></tr>"
    hours = "".join(f"<tr><td>{h:02d}:00</td><td>{sum(v.values())}</td><td>{v.get('PASS', 0)}</td>"
                    f"<td>{v.get('REJECT', 0)}</td><td>{v.get('FAULT', 0)}</td></tr>" for h, v in sorted(s["hourly"].items()))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(title)}</title>
<style>body{{font:14px Segoe UI,Arial,sans-serif;margin:28px;color:#111}}h1{{margin:0 0 4px}}h2{{margin:26px 0 6px;font-size:16px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #c8ccd2;padding:4px 8px;text-align:left;font-size:12.5px}}th{{background:#eef1f4}}
.k{{display:flex;gap:14px;margin:12px 0}}.k div{{border:1px solid #c8ccd2;padding:10px 18px;border-radius:6px}}.k b{{display:block;font-size:26px}}
.g b{{color:#15803d}}.r b{{color:#c81e1e}}.f b{{color:#b45309}}.bar{{height:14px;background:#c81e1e}}small{{color:#555}}
@media print{{body{{margin:10mm}}}}</style></head><body>
<h1>{e(title)}</h1><small>{e(label)} &middot; {_fmt_time(span[0])} to {_fmt_time(span[1])} &middot; generated {_fmt_time(time.time())}</small>
<div class="k"><div><b>{total}</b>bottles</div><div class="g"><b>{s['PASS']}</b>GOOD ({pct(s['PASS'])})</div>
<div class="r"><b>{s['REJECT']}</b>DEFECT ({pct(s['REJECT'])})</div><div class="f"><b>{s['FAULT']}</b>FAULT ({pct(s['FAULT'])})</div>
<div><b>{s.get('not_inspected', 0)}</b>not inspected</div><div><b>{s['alarms_raised']}</b>alarms raised</div></div>
<h2>Defects</h2><table><tr><th>Defect</th><th>Bottles</th><th></th></tr>{bars}</table>
<h2>Bottles per hour</h2><table><tr><th>Hour</th><th>Total</th><th>Good</th><th>Defect</th><th>Fault</th></tr>{hours or "<tr><td colspan=5>none</td></tr>"}</table>
<h2>Runs (START to STOP)</h2>{table(["Run", "Started", "Stopped", "Job", "Product", "Recipe", "Models", "PLC"], runs)}
<h2>Alarms</h2>{table(["Time", "Code", "Severity", "State", "Message", "Cause", "Count"], alarms)}
<h2>Latest bottles (up to 200)</h2>{table(cols, rows)}
<p><small>Source: production.db (SQLite). GOOD = PASS sent to the PLC, DEFECT = REJECT, FAULT = could not be judged.</small></p>
</body></html>"""


def export_report(store, path, span, title="Production report", label="") -> None:
    Path(path).write_text(report_html(store, span, title, label), encoding="utf-8")


def info(store) -> dict:
    """File location, size, row counts and date range, for the Database page header."""
    with store._lock:
        n = {t: store._db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
        lo, hi = store._db.execute("SELECT MIN(wall), MAX(wall) FROM inspections").fetchone()
    size = store.path.stat().st_size if store.path.exists() else 0
    return {"path": str(store.path), "size_kb": round(size / 1024, 1), "rows": n, "first": _fmt_time(lo), "last": _fmt_time(hi),
            "evidence_dir": str(store.folder / "evidence")}


def demo():
    import tempfile
    from line.machine_cycle import Bottle
    import numpy as np
    from line.production_store import ProductionStore
    with tempfile.TemporaryDirectory() as td:
        st = ProductionStore(Path(td))
        st.start_run("demo", "JOB-7", "250 ml", None, {"detection": "yolov8n"}, {"plc_t0_s": 1.5}, "SIMULATOR")
        now = time.time()
        for i, (fin, dfx) in enumerate((("PASS", []), ("REJECT", ["missing_cap", "tilt_cap"]), ("FAULT", [])), 1):
            b = Bottle(f"{i:06d}", i, 10.0 + i, now - 60 + i)
            b.final = b.decision = fin
            b.defects, b.command, b.plc_status, b.confidence = dfx, ("PASS" if fin == "PASS" else "REJECT"), "ACKED", 0.9
            b.thumb = np.zeros((20, 20, 3), np.uint8)
            b.timings = {"trigger_to_decision_ms": 87.0}
            st.record(b)
        st.flush()
        span = (now - 3600, now + 3600)
        cols, rows = rows_for(st, "inspections", span)
        assert cols[2] == "result" and [r[2] for r in rows] == ["FAULT", "DEFECT", "GOOD"], rows
        assert rows[1][3] == "Missing cap, Tilted cap" and rows[1][7] == "87.0", rows[1]
        assert [r[2] for r in rows_for(st, "inspections", span, result="DEFECT")[1]] == ["DEFECT"]
        assert len(rows_for(st, "inspections", span, text="tilted")[1]) == 1
        assert rows_for(st, "runs", span)[1][0][3] == "JOB-7" and "SIMULATOR" in rows_for(st, "runs", span)[1][0]
        assert not rows_for(st, "inspections", (0, 1))[1]
        p = Path(td) / "x.csv"
        assert export_csv(st, "inspections", p, span) == 3
        txt = p.read_text(encoding="utf-8-sig").splitlines()
        assert txt[0].startswith("time,bottle,result") and "DEFECT" in txt[2], txt
        rep = Path(td) / "r.html"
        export_report(st, rep, span, "Test report", "shift A")
        h = rep.read_text(encoding="utf-8")
        assert "Test report" in h and "Missing cap" in h and "<b>3</b>bottles" in h and "JOB-7" in h, h[:300]
        i = info(st)
        assert i["rows"]["inspections"] == 3 and i["rows"]["runs"] == 1 and i["path"].endswith("production.db"), i
        for t in TABLES:
            assert EXPLAIN[t]
        st.close()
    print("ok  production_export: plain-word rows (GOOD / DEFECT, readable defects), filters, CSV (Excel), printable HTML "
          "report, table info")


if __name__ == "__main__":
    demo()
