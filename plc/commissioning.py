"""Small PLC commissioning window. NOT a production dashboard: it exists to prove the link and
watch devices while the ladder is being tested.

    python -m plc.commissioning                  # against the ISPSoft simulator (127.0.0.1:10002)
    python -m plc.commissioning --selftest       # builds the window against a fake PLC, no hardware

Design rules (see SYSTEM_ROADMAP/PLC_COMMUNICATION.md):
  * All PLC calls run on one worker thread; the window only reads results. A PLC timeout cannot
    freeze the UI.
  * Polling is rate-limited (one snapshot in flight at a time) -- it cannot flood the PLC.
  * Nothing is forced. There is no output-forcing control at all. The only write control is the
    "command bit", enabled only if address_map.WRITE_ALLOWLIST names exactly one M/D bit.
  * Disconnected / FAULT is shown as such and values go stale ("--"): a dead link is never shown
    as a good value, and nothing is sent until you press Connect again.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import customtkinter as ctk

from . import address_map as AM
from .client import CONNECTED, FAULT, PLCClient, TcpTransport

GOOD, BAD, WARN, DIM, ON_COL, OFF_COL = "#15803d", "#b91c1c", "#b45309", "#5b6672", "#15803d", "#94a3b8"


class Commissioning(ctk.CTk):
    POLL_HZ = 4.0

    def __init__(self, plc: PLCClient):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("PLC commissioning - ISPSoft simulator (NOT a production dashboard)")
        self.geometry("980x640")
        self.plc = plc
        self.jobs: "queue.Queue" = queue.Queue()
        self.results: "queue.Queue" = queue.Queue()
        self.busy = False
        self.polling = False
        self.snapshot: dict | None = None
        self._stop = False
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()
        self._build()
        self.after(100, self._drain)
        self.after(int(1000 / self.POLL_HZ), self._poll)
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ---------------------------------------------------------------- UI
    def _build(self):
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=10, pady=8)
        self.state_lbl = ctk.CTkLabel(top, text="DISCONNECTED", font=("Segoe UI", 20, "bold"), text_color=DIM)
        self.state_lbl.pack(side="left", padx=12, pady=6)
        for text, cmd in (("Connect", self.do_connect), ("Disconnect", self.do_disconnect),
                          ("Read all", self.do_read), ("Heartbeat x20", self.do_heartbeat)):
            ctk.CTkButton(top, text=text, width=110, command=cmd).pack(side="left", padx=4)
        self.poll_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(top, text=f"auto-poll {self.POLL_HZ:g} Hz", variable=self.poll_var,
                        command=lambda: setattr(self, "polling", self.poll_var.get())).pack(side="left", padx=12)

        info = ctk.CTkFrame(self)
        info.pack(fill="x", padx=10)
        self.info_lbl = ctk.CTkLabel(info, text="", justify="left", anchor="w", font=("Consolas", 12))
        self.info_lbl.pack(fill="x", padx=10, pady=6)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=6)
        self.cells: dict = {}
        for col, (title, kind) in enumerate((("INPUTS (X)", "X"), ("INTERNAL (M)", "M"),
                                             ("OUTPUTS (Y)", "Y"), ("TIMERS (T, value)", "T"))):
            box = ctk.CTkFrame(body)
            box.grid(row=0, column=col, sticky="nsew", padx=4)
            body.grid_columnconfigure(col, weight=1)
            ctk.CTkLabel(box, text=title, font=("Segoe UI", 13, "bold")).pack(pady=(6, 2))
            for d in AM.group(kind):
                if d.name == AM.RUN_FLAG:
                    continue
                row = ctk.CTkFrame(box, fg_color="transparent")
                row.pack(fill="x", padx=8, pady=1)
                ctk.CTkLabel(row, text=d.name, width=44, anchor="w", font=("Consolas", 13, "bold")).pack(side="left")
                v = ctk.CTkLabel(row, text="--", width=54, corner_radius=6, fg_color=OFF_COL, text_color="white")
                v.pack(side="left", padx=4)
                note = f"@0x{d.address:04X}" + ("" if d.in_ladder_list else "  (not in ladder list)")
                ctk.CTkLabel(row, text=note, text_color=DIM, font=("Consolas", 10)).pack(side="left")
                self.cells[d.name] = v
        ctk.CTkLabel(body, text="Meanings are UNKNOWN: the ladder is not readable from here. Addresses are shown, not roles.",
                     text_color=WARN, font=("Segoe UI", 11)).grid(row=1, column=0, columnspan=4, pady=6)

        cmd = ctk.CTkFrame(self)
        cmd.pack(fill="x", padx=10, pady=(0, 8))
        self.cmd_bit = next(iter(AM.WRITE_ALLOWLIST)) if len(AM.WRITE_ALLOWLIST) == 1 else None
        msg = (f"Command bit: {self.cmd_bit}" if self.cmd_bit else
               "Command bit: NONE configured - needs the ladder (a safe M bit chosen by whoever has read it). "
               "No write control is available; outputs can never be forced from here.")
        ctk.CTkLabel(cmd, text=msg, text_color=DIM if not self.cmd_bit else GOOD, wraplength=640,
                     justify="left").pack(side="left", padx=10, pady=6)
        for text, val in (("Command ON", True), ("Command OFF / reset", False)):
            ctk.CTkButton(cmd, text=text, width=150, state="normal" if self.cmd_bit else "disabled",
                          command=lambda v=val: self.do_command(v)).pack(side="right", padx=4)
        self.msg_lbl = ctk.CTkLabel(self, text="", text_color=DIM, anchor="w")
        self.msg_lbl.pack(fill="x", padx=14, pady=(0, 6))

    # ---------------------------------------------------------------- worker (all PLC I/O)
    def _work(self):
        while not self._stop:
            try:
                name, fn = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.results.put((name, fn(), None))
            except Exception as e:                       # noqa: BLE001 - surfaced in the UI, never swallowed
                self.results.put((name, None, e))

    def submit(self, name, fn):
        self.busy = True
        self.jobs.put((name, fn))

    # ---------------------------------------------------------------- actions
    def do_connect(self):
        self.snapshot = None
        self.submit("connect", self.plc.reconnect)

    def do_disconnect(self):
        self.polling = False
        self.poll_var.set(False)
        self.submit("disconnect", self.plc.disconnect)

    def do_read(self):
        self.submit("read", self.plc.read_status)

    def do_heartbeat(self):
        def run():
            lat = []
            for _ in range(20):
                lat.append(self.plc.heartbeat()["latency_ms"])
            lat.sort()
            return f"heartbeat x20: mean {sum(lat) / 20:.1f} ms, p95 {lat[18]:.1f} ms, max {lat[-1]:.1f} ms, 0 failures"
        self.submit("heartbeat", run)

    def do_command(self, on: bool):
        if self.cmd_bit:
            self.submit("command", lambda: self.plc.write_bit(self.cmd_bit, on))

    def _poll(self):
        if self.polling and not self.busy and self.plc.is_connected():
            self.do_read()
        if not self._stop:
            self.after(int(1000 / self.POLL_HZ), self._poll)

    # ---------------------------------------------------------------- results -> widgets
    def _drain(self):
        try:
            while True:
                name, val, err = self.results.get_nowait()
                self.busy = False
                if err is not None:
                    self.msg_lbl.configure(text=f"{name} FAILED: {type(err).__name__}: {err}", text_color=BAD)
                    self.polling = False
                    self.poll_var.set(False)
                    self.snapshot = None                 # values are stale now: do not keep showing them
                elif name == "read":
                    self.snapshot = val
                    self.msg_lbl.configure(text="", text_color=DIM)
                elif name in ("heartbeat",):
                    self.msg_lbl.configure(text=val, text_color=GOOD)
                elif name == "connect":
                    self.msg_lbl.configure(text=f"connected, first heartbeat {val:.1f} ms", text_color=GOOD)
                    self.do_read()
                elif name == "command":
                    self.msg_lbl.configure(text=f"command bit written", text_color=GOOD)
        except queue.Empty:
            pass
        self._refresh()
        if not self._stop:
            self.after(100, self._drain)

    def _refresh(self):
        h = self.plc.health()
        colour = {CONNECTED: GOOD, FAULT: BAD}.get(h["state"], DIM)
        self.state_lbl.configure(text=h["state"], text_color=colour)
        age = "never" if h["age_s"] is None else f"{h['age_s']:.1f}s ago"
        lat = "--" if h["last_latency_ms"] is None else f"{h['last_latency_ms']:.1f} ms"
        run = "--" if not self.snapshot else ("RUN" if self.snapshot["plc_run"] else "NOT RUN")
        self.info_lbl.configure(text=(
            f"target   : {h['target']}   {h['transport']}   {h['protocol']}   station {h['station']}\n"
            f"last good reply: {age}   latency {lat}   PLC mode {run}   ok {h['ok']}  errors {h['errors']}  "
            f"consecutive {h['consecutive_errors']}\n"
            f"last error: {h['last_error'] or '-'}"))
        snap = self.snapshot if h["state"] == CONNECTED else None
        flat = {}
        if snap:
            for k in ("inputs", "internal", "outputs"):
                flat.update(snap[k])
        for name, w in self.cells.items():
            if snap is None:
                w.configure(text="--", fg_color=OFF_COL)
            elif name in snap["timers"]:
                w.configure(text=str(snap["timers"][name]), fg_color="#1d4ed8" if snap["timers"][name] else OFF_COL)
            elif name in flat:
                w.configure(text="ON" if flat[name] else "OFF", fg_color=ON_COL if flat[name] else OFF_COL)

    def close(self):
        self._stop = True
        self.polling = False
        try:
            self.plc.disconnect()                        # sends nothing; this tool never leaves a bit set
        finally:
            self.destroy()


def selftest() -> int:
    """Window against a FAKE PLC: builds, connects, reads, polls, shows FAULT on a dead link, closes."""
    from .test_simulation import FakePLC, _client
    fake = FakePLC()
    fake.bits[AM.address_of("M1")] = 1
    fake.regs[AM.address_of("T3")] = 9
    app = Commissioning(_client(fake, timeout=0.3))

    def pump(sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            app.update(); time.sleep(0.02)
    try:
        pump(0.3)
        assert app.state_lbl.cget("text") == "DISCONNECTED" and app.cells["M1"].cget("text") == "--"
        assert str(app.cmd_bit) == "None", "no command bit may be configured by default"
        app.do_connect(); pump(1.0)
        assert app.state_lbl.cget("text") == CONNECTED, app.state_lbl.cget("text")
        assert app.cells["M1"].cget("text") == "ON" and app.cells["M0"].cget("text") == "OFF"
        assert app.cells["T3"].cget("text") == "9"
        app.poll_var.set(True); app.polling = True
        n0 = fake.count; pump(1.2)
        assert 2 <= (fake.count - n0) / 5 <= 8, f"poll rate off: {(fake.count - n0)} transactions in 1.2 s"
        fake.mode = "silent"                              # the PLC stops answering
        pump(1.6)
        assert app.state_lbl.cget("text") == FAULT, app.state_lbl.cget("text")
        assert app.cells["M1"].cget("text") == "--", "stale value still shown after the link died"
        n1 = fake.count; pump(0.8)
        assert fake.count == n1, "commands kept flowing after FAULT"
        fake.mode = "ok"; app.do_connect(); pump(1.0)
        assert app.state_lbl.cget("text") == CONNECTED and app.cells["M1"].cget("text") == "ON"
        app.do_heartbeat(); pump(1.0)
        assert "heartbeat x20" in app.msg_lbl.cget("text"), app.msg_lbl.cget("text")
    finally:
        app.close()
        fake.close()
    print("ok  commissioning window (fake PLC): builds, connects, reads, rate-limited poll, FAULT shows stale '--' "
          "and stops sending, explicit reconnect, heartbeat, no command bit by default, clean close")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=10002)
    ap.add_argument("--station", type=int, default=1)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    Commissioning(PLCClient(TcpTransport(a.host, a.port), station=a.station)).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
