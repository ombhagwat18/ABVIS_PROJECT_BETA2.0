"""PLC commissioning window. Not a production dashboard: it proves the link and shows the
handshake while the ladder is being tested.

    python -m plc.commissioning                  # against the ISPSoft simulator (127.0.0.1:10002)
    python -m plc.commissioning --selftest       # builds the window against a FAKE PLC, no hardware

Design rules (see docs/roadmap/PLC_COMMUNICATION.md):
  * The window owns nothing: one PLCService (one worker thread, one PLC client) does all PLC I/O.
    The window only reads its cached snapshot, so a PLC timeout can never freeze the UI.
  * Two guarded commands only: PASS (M0) and REJECT (M1). They are enabled only when the link is
    CONNECTED, the PLC has an inspection pending (M2 = 1) and the "arm" box is ticked. There is no
    control that writes Y0, Y1, X or any other device, and no way to resend a command.
  * A command is shown as done only when the PLC acknowledged it (bit and M2 cleared).
  * Disconnected / FAULT / stale is shown as such and values go "--": a dead link is never shown as
    a good value, and nothing is sent until you press Connect again.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import customtkinter as ctk

from . import address_map as AM
from .client import CONNECTED, DISCONNECTED, FAULT, PLCClient, TcpTransport
from .service import ACKED, DEGRADED, T_PENDING, PLCService

GOOD, BAD, WARN, DIM, ON_COL, OFF_COL = "#15803d", "#b91c1c", "#b45309", "#5b6672", "#15803d", "#94a3b8"
SHOW = [("INPUTS (X)", "X", ["X0", "X1", "X2"]), ("INTERNAL (M)", "M", ["M0", "M1", "M2"]),
        ("OUTPUTS (Y)", "Y", ["Y0", "Y1"]), ("TIMERS / COUNTERS", "T", ["T0", "T1", "C0", "C1"])]
SHORT = {"X0": "sensor", "X1": "start", "X2": "stop", "M0": "PASS cmd", "M1": "REJECT cmd", "M2": "trigger",
         "Y0": "reject", "Y1": "conveyor", "T0": "delay", "T1": "pulse", "C0": "?", "C1": "?"}


class Commissioning(ctk.CTk):
    REFRESH_MS = 150

    def __init__(self, svc: PLCService):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("PLC commissioning (NOT a production dashboard)")
        self.geometry("1020x700")
        self.svc = svc
        self.svc.start()
        self.cmd_result = None
        self.cmd_busy = False
        self.op_msg, self.op_ok = "", True
        self._lock = threading.Lock()
        self._closed = False
        self._build()
        self.after(self.REFRESH_MS, self._refresh)
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ---------------------------------------------------------------- UI
    def _build(self):
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=10, pady=8)
        self.state_lbl = ctk.CTkLabel(top, text="DISCONNECTED", font=("Segoe UI", 20, "bold"), text_color=DIM)
        self.state_lbl.pack(side="left", padx=12, pady=6)
        ctk.CTkButton(top, text="Connect", width=110, command=lambda: self._op("connect", self.svc.connect)).pack(side="left", padx=4)
        ctk.CTkButton(top, text="Disconnect", width=110, command=lambda: self._op("disconnect", self.svc.disconnect)).pack(side="left", padx=4)
        self.info_lbl = ctk.CTkLabel(self, text="", justify="left", anchor="w", font=("Consolas", 12))
        self.info_lbl.pack(fill="x", padx=20, pady=4)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="x", padx=10, pady=4)
        self.cells: dict = {}
        for col, (title, _kind, names) in enumerate(SHOW):
            box = ctk.CTkFrame(body)
            box.grid(row=0, column=col, sticky="nsew", padx=4)
            body.grid_columnconfigure(col, weight=1)
            ctk.CTkLabel(box, text=title, font=("Segoe UI", 13, "bold")).pack(pady=(6, 2))
            for n in names:
                row = ctk.CTkFrame(box, fg_color="transparent")
                row.pack(fill="x", padx=8, pady=1)
                ctk.CTkLabel(row, text=n, width=40, anchor="w", font=("Consolas", 13, "bold")).pack(side="left")
                v = ctk.CTkLabel(row, text="--", width=52, corner_radius=6, fg_color=OFF_COL, text_color="white")
                v.pack(side="left", padx=4)
                ctk.CTkLabel(row, text=SHORT[n], text_color=DIM, font=("Consolas", 11)).pack(side="left")
                self.cells[n] = v

        cmd = ctk.CTkFrame(self)
        cmd.pack(fill="x", padx=10, pady=6)
        self.trig_lbl = ctk.CTkLabel(cmd, text="inspection trigger: none", font=("Segoe UI", 13, "bold"), anchor="w")
        self.trig_lbl.pack(fill="x", padx=10, pady=(6, 2))
        row = ctk.CTkFrame(cmd, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=4)
        self.arm_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(row, text="arm commands", variable=self.arm_var).pack(side="left", padx=(0, 14))
        self.btn_pass = ctk.CTkButton(row, text="PASS command (M0)", width=170, fg_color=GOOD, state="disabled",
                                      command=lambda: self._send("PASS"))
        self.btn_pass.pack(side="left", padx=4)
        self.btn_rej = ctk.CTkButton(row, text="REJECT command (M1)", width=180, fg_color=BAD, state="disabled",
                                     command=lambda: self._send("REJECT"))
        self.btn_rej.pack(side="left", padx=4)
        self.cmd_lbl = ctk.CTkLabel(cmd, text="", anchor="w", justify="left", font=("Consolas", 12), wraplength=960)
        self.cmd_lbl.pack(fill="x", padx=10, pady=(2, 6))
        ctk.CTkLabel(self, text=("Y0/Y1 are never written from here: the PLC owns conveyor and reject timing. A command is "
                                 "sent once, only for a pending trigger, and is never retried."),
                     text_color=WARN, font=("Segoe UI", 11), anchor="w").pack(fill="x", padx=20)
        self.msg_lbl = ctk.CTkLabel(self, text="", text_color=DIM, anchor="w")
        self.msg_lbl.pack(fill="x", padx=20, pady=2)
        self.log = ctk.CTkTextbox(self, height=170, font=("Consolas", 11))
        self.log.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    # ---------------------------------------------------------------- actions (never on the UI thread)
    def _op(self, name, fn):
        def run():
            try:
                fn()
                msg, ok = f"{name} ok", True
            except Exception as e:                       # noqa: BLE001 - surfaced, never swallowed
                msg, ok = f"{name} FAILED: {type(e).__name__}: {e}", False
            with self._lock:
                self.op_msg, self.op_ok = msg, ok
        threading.Thread(target=run, daemon=True).start()

    def _send(self, verdict: str):
        trig = self.svc.current_trigger()
        if self.cmd_busy or trig is None or trig.state != T_PENDING or not self.arm_var.get():
            return
        self.cmd_busy = True
        self.arm_var.set(False)                           # one press = one command; re-arm for the next
        tid = trig.id

        def run():
            res = self.svc.submit_result(tid, verdict)
            with self._lock:
                self.cmd_result = res
        threading.Thread(target=run, daemon=True).start()

    # ---------------------------------------------------------------- refresh from the cached state
    def _refresh(self):
        if self._closed:
            return
        h = self.svc.health_check()
        st = h["link_state"]
        self.state_lbl.configure(text=st, text_color={CONNECTED: GOOD, DEGRADED: WARN, FAULT: BAD}.get(st, DIM))
        age = "never" if h["age_s"] is None else f"{h['age_s']:.1f}s ago"
        lat = "--" if h["last_latency_ms"] is None else f"{h['last_latency_ms']:.1f} ms"
        lw = "--" if not h["last_ok_wall"] else time.strftime("%H:%M:%S", time.localtime(h["last_ok_wall"]))
        self.info_lbl.configure(text=(
            f"target {h['target']}   {h['transport']}   {h['protocol']}   station {h['station']}\n"
            f"latency {lat}   last good reply {lw} ({age})   ok {h['ok']}  errors {h['errors']}  consecutive {h['consecutive_errors']}\n"
            f"last error: {h['last_error'] or '-'}"))
        snap = self.svc.snapshot() if st == CONNECTED else None
        flat = {}
        if snap:
            for k in ("inputs", "internal", "outputs", "timers", "counters"):
                flat.update(snap[k])
        for name, w in self.cells.items():
            if name not in flat:
                w.configure(text="--", fg_color=OFF_COL)
            elif name[0] in "TC":
                w.configure(text=str(flat[name]), fg_color="#1d4ed8" if flat[name] else OFF_COL)
            else:
                w.configure(text="ON" if flat[name] else "OFF", fg_color=ON_COL if flat[name] else OFF_COL)
        trig = self.svc.current_trigger()
        if st != CONNECTED or trig is None:
            self.trig_lbl.configure(text="inspection trigger: none" if st == CONNECTED else f"inspection trigger: unknown (link {st})")
        else:
            late = "  OVERDUE" if trig.overdue else ""
            flag = "  (seen right after connect: M2 may be stale)" if trig.after_reconnect else ""
            self.trig_lbl.configure(text=f"inspection trigger #{trig.id}: {trig.state}{late}{flag}")
        can = st == CONNECTED and trig is not None and trig.state == T_PENDING and self.arm_var.get() and not self.cmd_busy
        for b in (self.btn_pass, self.btn_rej):
            b.configure(state="normal" if can else "disabled")
        with self._lock:
            res, self.cmd_result = self.cmd_result, None
            msg, ok = self.op_msg, self.op_ok
        if res is not None:
            self.cmd_busy = False
            tail = f"write->response {res.write_ms:.1f} ms, ack {res.ack_ms:.1f} ms" if res.status == ACKED else res.detail
            self.cmd_lbl.configure(text=f"{res.command} for trigger #{res.trigger_id}: {res.status}  {tail}",
                                   text_color=GOOD if res.status == ACKED else BAD)
        self.msg_lbl.configure(text=msg, text_color=DIM if ok else BAD)
        evs = self.svc.events()[-14:]
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        for e in evs:
            lat = f" {e.latency_ms:.1f}ms" if e.latency_ms is not None else ""
            self.log.insert("end", f"{time.strftime('%H:%M:%S', time.localtime(e.wall))} {e.event:<20}"
                                   f"{e.device:<4}{e.command:<7}{('#' + str(e.trigger_id)) if e.trigger_id else '':<5}{lat} "
                                   f"{e.ack} {e.error}\n")
        self.log.configure(state="disabled")
        self.after(self.REFRESH_MS, self._refresh)

    def close(self):
        self._closed = True
        try:
            self.svc.stop()                              # sends nothing; the PLC clears M0/M1 itself
        finally:
            self.destroy()


def selftest() -> int:
    """Window against a FAKE PLC + FakeLadder: builds, connects, shows a trigger, gated PASS/REJECT, FAULT."""
    from .test_simulation import FakeLadder, FakePLC, _client
    fake = FakePLC()
    lad = FakeLadder(fake)
    svc = PLCService(_client(fake, timeout=0.3), poll_s=0.02, status_period_s=0.1)
    app = Commissioning(svc)

    def pump(sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            app.update(); time.sleep(0.02)

    def wait(cond, sec=3.0):
        t0 = time.time()
        while time.time() - t0 < sec:
            app.update(); time.sleep(0.02)
            if cond():
                return True
        return False
    try:
        pump(0.4)
        assert app.state_lbl.cget("text") == DISCONNECTED and app.cells["M2"].cget("text") == "--"
        assert not hasattr(app, "do_force") and "Y0" not in str(AM.WRITE_ALLOWLIST)
        app._op("connect", svc.connect)
        assert wait(lambda: app.state_lbl.cget("text") == CONNECTED), app.state_lbl.cget("text")
        assert wait(lambda: app.cells["M2"].cget("text") == "OFF"), "snapshot never arrived"
        assert str(app.btn_pass.cget("state")) == "disabled", "command enabled with no trigger"
        lad.trigger()
        assert wait(lambda: "PENDING" in app.trig_lbl.cget("text"))
        assert wait(lambda: app.cells["M2"].cget("text") == "ON")
        pump(0.3)
        assert str(app.btn_rej.cget("state")) == "disabled", "command enabled without arming"
        app._send("PASS"); pump(0.3)
        assert not fake.writes, "command sent without arming"
        app.arm_var.set(True)
        assert wait(lambda: str(app.btn_pass.cget("state")) == "normal")
        app._send("REJECT")
        assert wait(lambda: "ACKED" in app.cmd_lbl.cget("text")), app.cmd_lbl.cget("text")
        assert len([w for w in fake.writes if w[1] == 1]) == 1 and fake.writes[0][0] == AM.address_of("M1")
        assert not app.arm_var.get(), "arm box should reset after one command"
        assert wait(lambda: [v for _, v in lad.y0_log] == [1, 0], 4.0), lad.y0_log   # the fake ladder ran T0 -> Y0 -> T1
        assert wait(lambda: app.cells["M2"].cget("text") == "OFF")
        fake.mode = "silent"
        assert wait(lambda: app.state_lbl.cget("text") == FAULT, 4.0), app.state_lbl.cget("text")
        assert app.cells["M0"].cget("text") == "--", "stale value still shown after the link died"
        n = fake.count; pump(0.8)
        assert fake.count == n, "traffic kept flowing after FAULT"
        fake.mode = "ok"; app._op("connect", svc.connect)
        assert wait(lambda: app.state_lbl.cget("text") == CONNECTED)
        assert len([w for w in fake.writes if w[1] == 1]) == 1, "reconnect resent a command"
    finally:
        app.close(); lad.stop(); fake.close()
    print("ok  commissioning window (FAKE PLC): builds, connects, M2/trigger shown, PASS/REJECT gated by trigger+arm, "
          "one press -> one write -> ACKED, no Y/other write controls, FAULT shows '--' and stops sending, "
          "reconnect does not replay, clean close")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=10002)
    ap.add_argument("--station", type=int, default=1)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    Commissioning(PLCService(PLCClient(TcpTransport(a.host, a.port), station=a.station))).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
