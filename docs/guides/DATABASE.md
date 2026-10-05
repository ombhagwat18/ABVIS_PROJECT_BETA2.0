# The production database

**Which database?** SQLite. It is built into Python, needs no server and no installation, is a single file, and
survives a power cut (every write is a transaction). Any tool can open it (DB Browser for SQLite, Python, or Excel
through the CSV export).

**Where?** `projects/<project>/production/production.db`, one per product project. Evidence pictures are in
`projects/<project>/production/evidence/<date>/`. The old daily CSV (`production/<date>.csv`) is still written too.
These files are not in Git (back them up by copying the `production` folder).

**In the app:** Database page (also reachable by the operator). Pick *Bottles / Alarms / Runs*, a *range*
(today, 7 / 30 days, all, shift A/B/C), a *result* filter and a search text. Click a column title to sort. The right
panel explains every column. **Export CSV (Excel)** writes exactly the rows shown. **Production report (print /
PDF)** writes one page with totals, defect counts, bottles per hour, runs, alarms and the latest bottles; open it in
the browser and Print -> Save as PDF.

## Tables

| Table | One row per | Main columns |
|---|---|---|
| `runs` | START INSPECTION | run id, started / stopped, job, product, recipe fingerprint, model ids, SIMULATOR or REAL PLC, the line settings |
| `inspections` | finished bottle | time, bottle id, GOOD / DEFECT / FAULT, defects, confidence, command sent to the PLC, PLC answer, timings, evidence picture, run id, the full record as JSON |
| `alarms` | alarm change | time, code, severity, state (active / acknowledged / cleared), message, cause, count |

## When rows are written

- START INSPECTION -> a `runs` row, with the models and recipe that will judge the bottles.
- A bottle ends (PASS, REJECT or FAULT) -> an `inspections` row; a picture is kept by the evidence policy
  (Settings: NONE / ALL / REJECT_ONLY / FAULT_ONLY / REJECT_AND_FAULT / SAMPLE:n).
- An alarm is raised, acknowledged or cleared -> an `alarms` row.

Rows are only ever added. The application never deletes production history.

## Words on screen vs. in the PLC

The PLC and the code say PASS / REJECT / FAULT. The screens say **GOOD / DEFECT / FAULT** with the defect named
("DEFECT: Missing cap"). The database stores the PLC words in `final`; the Database page and exports show the screen
words.

## Reading it yourself

```python
import sqlite3
db = sqlite3.connect("projects/om_bottle/production/production.db")
for row in db.execute("SELECT inspection_id, final, defects FROM inspections ORDER BY rowid DESC LIMIT 10"):
    print(row)
```

Code: `production_store.py` (writes), `production_export.py` (reads, CSV, report), `hmi.DatabaseTab` (page).
