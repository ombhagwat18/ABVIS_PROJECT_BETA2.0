"""PLCService: the ONE owner of the PLC link inside the application, and the boundary the future
inspection engine talks to. It knows nothing about cameras, YOLO or images.

    PLC ladder:  X0 -> M2 = 1                          (bottle ready: "inspect now")
    Application: trig = svc.wait_for_trigger()         (event; each trigger is delivered once)
                 ... inspect ...
                 res  = svc.send_pass(trig.id)  /  svc.send_reject(trig.id)
    PLC ladder:  M0|M1 consumed, resets itself and M2; REJECT then runs T0 -> Y0 -> T1 -> Y0 off

Rules enforced here (each is tested in plc/test_simulation.py and plc/handshake_test.py):
  * Exactly one worker thread performs every PLC transaction, so two threads can never write M0/M1
    at once and the GUI never blocks on the PLC. Public methods enqueue a job and wait for it.
  * At-most-once commands. A trigger accepts ONE answer; the "answered" flag is set BEFORE any byte
    is sent. A command is never retried, not even after a timeout or a lost link. If the outcome is
    unknown the result says so (WRITE_FAILED / NOT_ACKED / ACK_LOST) and the application must decide.
  * Preconditions are re-read from the PLC immediately before the write: M2 = 1, M0 = 0, M1 = 0.
  * Y0/Y1 are never written (see address_map). The PLC owns conveyor, reject delay and pulse timing.
  * A command is only successful when the PLC ACKNOWLEDGES it: the command bit and M2 return to 0.
  * No automatic reconnect. After a fault, connect() is an explicit call; a trigger seen again after
    a reconnect is a NEW trigger flagged `after_reconnect`.
  * Link states: CONNECTED / DEGRADED (connected but no good reply for `stale_after_s`) / FAULT /
    DISCONNECTED. Never "healthy" without a recent good reply.
  * Every step is recorded as a PLCEvent (timestamp, event, device, command, ack, latency, error,
    trigger id) so a later trace/DB layer can attach them to an inspection cycle.
"""
from __future__ import annotations

import collections
import itertools
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from . import address_map as AM
from .client import CONNECTED, DISCONNECTED, FAULT, PLCClient
from .protocol import PLCError

PASS, REJECT = "PASS", "REJECT"
DEGRADED = "DEGRADED"

# command result status
ACKED = "ACKED"                  # PLC consumed the command (command bit and M2 cleared)
REFUSED = "REFUSED"              # nothing was sent (bad trigger, duplicate, precondition, link down)
WRITE_FAILED = "WRITE_FAILED"    # the write failed; it MAY or may not have reached the PLC. Not retried.
NOT_ACKED = "NOT_ACKED"          # written, link OK, but the PLC did not consume it in time. Not retried.
ACK_LOST = "ACK_LOST"            # written, then the link failed before the ack was seen. Not retried.

# trigger states
T_PENDING, T_ANSWERED, T_CANCELLED, T_LOST = "PENDING", "ANSWERED", "CANCELLED", "LOST"


class PLCLink(Protocol):
    """The stable high-level surface the inspection application depends on. Simulator and the
    future physical PLC differ only in the Transport underneath PLCClient."""
    def connect(self) -> float: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def read_bit(self, device: str) -> bool: ...
    def read_word(self, device: str) -> int: ...
    def read_status(self) -> dict: ...
    def wait_for_trigger(self, timeout: Optional[float] = None) -> "Optional[Trigger]": ...
    def send_pass(self, trigger_id: int) -> "CommandResult": ...
    def send_reject(self, trigger_id: int) -> "CommandResult": ...
    def health_check(self) -> dict: ...


@dataclass
class Trigger:
    id: int
    seen_mono: float
    seen_wall: float
    after_reconnect: bool = False      # first observation was right after (re)connect: M2 may be stale
    state: str = T_PENDING
    overdue: bool = False
    note: str = ""


@dataclass
class CommandResult:
    trigger_id: int
    command: str                       # PASS / REJECT
    status: str
    detail: str = ""
    device: str = ""
    write_sent: bool = False           # True if a write request was put on the wire
    write_ms: Optional[float] = None   # request -> Modbus response
    ack_ms: Optional[float] = None     # write started -> command bit and M2 observed cleared
    cmd_seen_on: Optional[bool] = None # was the command bit ever observed ON (the PLC may clear it within a scan)
    cmd_held: bool = False             # ACKED with the bit still ON: the ladder holds M1 for T0 + T1
    before: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == ACKED


@dataclass
class PLCEvent:
    ts: float                          # time.monotonic()
    wall: float                        # time.time(), display only
    event: str
    device: str = ""
    command: str = ""
    ack: str = ""
    latency_ms: Optional[float] = None
    error: str = ""
    trigger_id: Optional[int] = None


class PLCService:
    def __init__(self, client: PLCClient, poll_s: float = 0.05, status_period_s: float = 0.25,
                 stale_after_s: float = 1.0, trigger_overdue_s: float = 5.0, ack_timeout_s: float = 1.0,
                 ack_poll_s: float = 0.0, watch_x0: bool = False):
        self.client = client
        self.poll_s, self.status_period_s = poll_s, status_period_s
        self.stale_after_s, self.trigger_overdue_s = stale_after_s, trigger_overdue_s
        self.ack_timeout_s, self.ack_poll_s = ack_timeout_s, ack_poll_s
        self._jobs: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()                  # guards the small shared state below
        self._events: "collections.deque" = collections.deque(maxlen=5000)
        self._listeners: list = []
        self._trigger: Optional[Trigger] = None        # current trigger (PENDING or ANSWERED until M2 clears)
        self._delivered: "queue.Queue" = queue.Queue()
        self._m2_prev: Optional[bool] = None
        self._after_reconnect = False
        self._unresolved_command = False               # a command whose outcome is unknown
        self._snapshot: Optional[dict] = None
        self._snapshot_mono: Optional[float] = None
        self._last_status = 0.0
        self._last_poll = 0.0
        self._cmd_bits = (AM.PASS_BIT, AM.REJECT_BIT)
        self._prev_dev: dict = {}                      # last observed value per device, for DEVICE_CHANGE events
        # Bottle accounting (opt-in: one more transaction per poll). X0 is read BEFORE M2 in every poll and
        # the ladder sets M2 in the scan X0 rises, so when an X0 rise is read, M2 must have risen since the
        # previous X0 read (or in this one). If not, the PLC could not trigger for this bottle (M2 already ON,
        # or held reset by M1 during a reject cycle).
        self.watch_x0 = watch_x0
        self._x0_prev: Optional[bool] = None
        self._m2_rose = False                          # M2 rose since the last observation that read X0
        self.untriggered = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="plc-service", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker and close the link. Sends nothing; a command bit is never left 'owned'
        by Python because the PLC clears it itself."""
        self._stop.set()
        t = self._thread
        if t and t is not threading.current_thread():
            t.join(timeout=self.client.timeout + 2.5)
        self.client.disconnect()
        self._event("STOPPED")

    shutdown = stop

    def add_listener(self, fn: Callable[[PLCEvent], None]) -> None:
        """fn is called on the service thread for every event: keep it fast and non-blocking."""
        self._listeners.append(fn)

    # ------------------------------------------------------------------ job plumbing
    def _call(self, fn, timeout: Optional[float] = None):
        """Run fn on the service thread and return its result (re-raises its exception)."""
        if self._thread is None or not self._thread.is_alive():
            raise PLCError("PLCService is not running (call start())")
        if threading.current_thread() is self._thread:
            return fn()
        done, box = threading.Event(), {}
        self._jobs.put((fn, done, box))
        if not done.wait(timeout if timeout is not None else self.client.timeout * 8 + 10):
            raise PLCError("PLCService job did not finish in time")
        if "err" in box:
            raise box["err"]
        return box["val"]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                fn, done, box = self._jobs.get(timeout=self.poll_s)
            except queue.Empty:
                self._poll_cycle()
                continue
            try:
                box["val"] = fn()
            except BaseException as e:               # noqa: BLE001 - handed back to the caller
                box["err"] = e
            finally:
                done.set()
            # Back-to-back jobs must not starve trigger polling: an M2 fall/rise missed here would
            # hide the next trigger (found on the simulator, 2026-10-03).
            if time.monotonic() - self._last_poll >= self.poll_s:
                self._poll_cycle()
        # fail anything still queued so no caller hangs
        while True:
            try:
                _, done, box = self._jobs.get_nowait()
            except queue.Empty:
                break
            box["err"] = PLCError("PLCService stopped")
            done.set()

    # ------------------------------------------------------------------ events
    def _event(self, event: str, **kw) -> PLCEvent:
        ev = PLCEvent(time.monotonic(), time.time(), event, **kw)
        with self._lock:
            self._events.append(ev)
        for fn in list(self._listeners):
            try:
                fn(ev)
            except Exception:                         # noqa: BLE001 - a listener must not break the link
                pass
        return ev

    def events(self, since: Optional[float] = None) -> list:
        with self._lock:
            return [e for e in self._events if since is None or e.ts > since]

    # ------------------------------------------------------------------ state
    def link_state(self) -> str:
        s = self.client.state
        if s == CONNECTED:
            age = self.client.age_s()
            return DEGRADED if age is None or age > self.stale_after_s else CONNECTED
        return s                                      # FAULT or DISCONNECTED

    def is_connected(self) -> bool:
        return self.link_state() == CONNECTED

    def health_check(self) -> dict:
        """Cached health: never touches the PLC, never says 'ok' unless a recent reply proves it."""
        h = self.client.health()
        h["link_state"] = self.link_state()
        h["stale_after_s"] = self.stale_after_s
        with self._lock:
            tr = self._trigger
            h["trigger"] = None if tr is None else {"id": tr.id, "state": tr.state, "overdue": tr.overdue}
            h["unresolved_command"] = self._unresolved_command
        return h

    def snapshot(self) -> Optional[dict]:
        """Last full status read by the worker, with its age; None if never read or the link is not
        CONNECTED (values from a dead link are not returned)."""
        with self._lock:
            snap, ts = self._snapshot, self._snapshot_mono
        if snap is None or self.client.state != CONNECTED:
            return None
        out = dict(snap)
        out["age_s"] = time.monotonic() - ts
        return out

    # ------------------------------------------------------------------ connection
    def connect(self) -> float:
        """Open the link and prove the PLC answers. Explicit; commands are never replayed."""
        def job():
            with self._lock:
                self._snapshot = None
            self._prev_dev = {}
            was_fault = self.client.state == FAULT
            try:
                lat = self.client.reconnect()
            except PLCError as e:
                self._event("CONNECT_FAILED", error=f"{type(e).__name__}: {e}")
                raise
            self._m2_prev = None                      # first poll after connect classifies M2 afresh
            self._x0_prev = None                      # ...and X0: no bottle is inferred across a reconnect
            self._after_reconnect = True
            self._event("CONNECTED", latency_ms=lat, ack="after fault" if was_fault else "")
            return lat
        return self._call(job)

    def reconfigure(self, transport, station: int = 1, target: str = "") -> None:
        """Point the service at a different link (simulator TCP <-> physical serial). The old link is
        closed first; nothing is sent, nothing is replayed, and connect() stays an explicit call."""
        def job():
            self.client.disconnect()
            self._mark_trigger_lost("link reconfigured")
            self.client.transport, self.client.station = transport, int(station)
            if target:
                self.client.target = target
            self.client.last_error = None
            with self._lock:
                self._snapshot = None
            self._prev_dev = {}
            self._event("RECONFIGURED", ack=f"{transport.description} station {station}")
        if self._thread is not None and self._thread.is_alive():
            self._call(job)
        else:
            job()

    def link_test(self, n: int = 10) -> dict:
        """n read-only heartbeats (the RUN relay). Returns replies, latency stats and the RUN state; a
        failure raises and leaves the link FAULT like any other read."""
        def job():
            lat, run = [], None
            for _ in range(n):
                hb = self._guard(self.client.heartbeat)
                lat.append(hb["latency_ms"]); run = hb["plc_run"]
            lat.sort()
            return {"replies": len(lat), "median_ms": lat[len(lat) // 2], "max_ms": lat[-1], "plc_run": run}
        return self._call(job)

    def disconnect(self) -> None:
        def job():
            self.client.disconnect()
            self._mark_trigger_lost("disconnect")
            self._event("DISCONNECTED")
        self._call(job)

    # ------------------------------------------------------------------ plain reads (serialised)
    def read_bit(self, device: str) -> bool:
        return self._call(lambda: self._guard(lambda: self.client.read_bit(device)))

    def read_word(self, device: str) -> int:
        return self._call(lambda: self._guard(lambda: self.client.read_word(device)))

    def read_status(self) -> dict:
        return self._call(lambda: self._guard(self._read_status_now))

    def sample(self, bits=(), words=()) -> dict:
        """One serialised read of several bit devices and word values: {NAME: value}. For test
        harnesses and the commissioning view; it never writes."""
        def job():
            out = {}
            if bits:
                out.update(self.client.read_many(list(bits)))
            if words:
                out.update(self.client.read_words_many(list(words)))
            return out
        return self._call(lambda: self._guard(job))

    def simulator_test_write(self, device: str, value: bool) -> dict:
        """SIMULATOR-ONLY TEST STIMULUS -- NOT a production path. Writes one of X0/X1/X2 (simulated operator /
        sensor inputs) or M0/M1/M2 outside the trigger/command protocol. Refused unless the transport is a
        TCP link to a loopback address (the ISPSoft simulator). X/Y and every other device are refused.
        Runs on the service thread, is logged as SIM_TEST_WRITE, and is never retried."""
        dev = device.strip().upper()
        host = getattr(self.client.transport, "host", None)
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise AM.AddressError(f"simulator_test_write refused: transport {self.client.transport.description!r} "
                                  f"is not a loopback simulator")
        if dev not in AM.SIM_TEST_BITS:
            raise AM.AddressError(f"simulator_test_write refused: {device!r} (only {sorted(AM.SIM_TEST_BITS)})")

        def job():
            t0 = time.perf_counter()
            self._event("SIM_TEST_WRITE", device=dev, ack=f"SIMULATOR-ONLY stimulus: {dev} <- {int(bool(value))}")
            with AM.sim_test_allow(dev):
                self._guard(lambda: self.client.write_bit(dev, bool(value)))
            ms = (time.perf_counter() - t0) * 1000
            self._event("SIM_TEST_WRITTEN", device=dev, latency_ms=ms)
            return {"device": dev, "value": int(bool(value)), "write_ms": ms}
        return self._call(job)

    def _guard(self, fn):
        """Run a PLC read; on a link failure make the loss visible (event + trigger bookkeeping)."""
        try:
            return fn()
        except PLCError as e:
            self._on_link_error(e)
            raise

    def _on_link_error(self, e: Exception) -> None:
        if self.client.state == FAULT:
            self._event("FAULT", error=f"{type(e).__name__}: {e}")
            self._mark_trigger_lost("link fault")

    def _read_status_now(self) -> dict:
        snap = self.client.read_status()
        snap["m2"] = bool(self.client.read_bit(AM.TRIGGER_BIT)) if AM.TRIGGER_BIT not in snap["internal"] \
            else bool(snap["internal"][AM.TRIGGER_BIT])
        with self._lock:
            self._snapshot, self._snapshot_mono = snap, time.monotonic()
        self._last_status = time.monotonic()
        self._device_changes(snap)
        self._observe(dict(snap["internal"], X0=snap["inputs"]["X0"]) if self.watch_x0 else snap["internal"])
        return snap

    def _device_changes(self, snap: dict) -> None:
        """One DEVICE_CHANGE event per observed change of X/M/Y (and T running/idle). Only values the PLC
        actually returned; the time is when this poll saw the change, not when it happened."""
        cur = {}
        for k in ("inputs", "internal", "outputs"):
            cur.update(snap[k])
        for n, v in snap["timers"].items():
            cur[n] = 1 if v else 0
        for n, v in cur.items():
            old = self._prev_dev.get(n)
            if old is not None and old != v:
                self._event("DEVICE_CHANGE", device=n, ack="ON" if v else "OFF",
                            latency_ms=float(snap["timers"][n]) if n in snap["timers"] else None)
        for n, v in snap.get("counters", {}).items():          # counters: report the new value, not ON/OFF
            old = self._prev_dev.get(n)
            if old is not None and old != v:
                self._event("DEVICE_CHANGE", device=n, ack=str(v))
            cur[n] = v
        self._prev_dev = cur

    @property
    def simulator_mode(self) -> bool:
        """True only for a TCP link to a loopback address (the ISPSoft simulator)."""
        return getattr(self.client.transport, "host", None) in ("127.0.0.1", "localhost", "::1")

    # ------------------------------------------------------------------ trigger tracking
    def _observe(self, internal: dict) -> None:
        """Feed one reading of M0/M1/M2 (and X0, when watched) into the trigger state machine
        (service thread only)."""
        m2 = bool(internal.get(AM.TRIGGER_BIT, 0))
        now = time.monotonic()
        with self._lock:
            tr = self._trigger
        rose_now = bool(m2 and not self._m2_prev)
        x0 = internal.get("X0")
        if x0 is None:                                # an observation without X0 (e.g. a command's ack read)
            self._m2_rose = self._m2_rose or rose_now
        else:
            rising = bool(x0) and self._x0_prev is False
            # X0 was read before M2 here, so an M2 rise in THIS observation can explain this X0 rise or the
            # next one; an earlier rise explains only this one. Each M2 rise accounts for one X0 rise.
            if rising and not (self._m2_rose or rose_now):
                why = ("M1 ON: the reject cycle holds M2 reset (ladder net 5)" if internal.get(AM.REJECT_BIT)
                       else "M2 already ON: the previous bottle was not answered yet" if self._m2_prev
                       else "X0 rose but the ladder did not set M2")
                self.untriggered += 1
                self._event("BOTTLE_UNTRIGGERED", device="X0", error=why)
            self._m2_rose = rose_now and not rising
            self._x0_prev = bool(x0)
        if m2:
            if not self._m2_prev:                     # rising edge (or first reading after connect)
                trig = Trigger(next(self._ids), now, time.time(), after_reconnect=bool(self._after_reconnect))
                if self._unresolved_command:
                    trig.note = "a command from before the fault had an unknown outcome"
                with self._lock:
                    self._trigger = trig
                self._delivered.put(trig)
                self._event("TRIGGER", device=AM.TRIGGER_BIT, trigger_id=trig.id,
                            ack="after reconnect: M2 may be stale" if trig.after_reconnect else "")
            elif tr is not None and tr.state == T_PENDING and not tr.overdue \
                    and now - tr.seen_mono > self.trigger_overdue_s:
                tr.overdue = True
                self._event("TRIGGER_OVERDUE", device=AM.TRIGGER_BIT, trigger_id=tr.id,
                            error=f"no result for {self.trigger_overdue_s:g}s")
        else:
            if tr is not None:
                if tr.state == T_PENDING:
                    tr.state = T_CANCELLED
                    self._event("TRIGGER_CANCELLED", device=AM.TRIGGER_BIT, trigger_id=tr.id,
                                error="M2 cleared without a command from Python")
                with self._lock:
                    if self._trigger is tr:
                        self._trigger = None
            self._unresolved_command = False              # M2 low: nothing outstanding any more
        self._m2_prev = m2
        self._after_reconnect = False

    def _mark_trigger_lost(self, why: str) -> None:
        with self._lock:
            tr = self._trigger
        if tr is not None and tr.state == T_PENDING:
            tr.state = T_LOST
            self._event("TRIGGER_LOST", device=AM.TRIGGER_BIT, trigger_id=tr.id, error=why)
        with self._lock:
            self._trigger = None
        self._m2_prev = None
        self._x0_prev = None

    def _poll_cycle(self) -> None:
        self._last_poll = time.monotonic()
        if self.client.state != CONNECTED:
            return
        try:
            if time.monotonic() - self._last_status >= self.status_period_s:
                self._read_status_now()
            else:
                self._observe(self.client.read_many((["X0"] if self.watch_x0 else []) + [AM.TRIGGER_BIT, *self._cmd_bits]))
        except PLCError as e:
            self._on_link_error(e)

    def current_trigger(self) -> Optional[Trigger]:
        with self._lock:
            return self._trigger

    def wait_for_trigger(self, timeout: Optional[float] = None) -> Optional[Trigger]:
        """Block until a NEW inspection trigger is observed (M2 rising). Each trigger is delivered
        once. Returns None on timeout. A trigger that has already been cancelled/lost is skipped."""
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            left = None if end is None else max(0.0, end - time.monotonic())
            try:
                trig = self._delivered.get(timeout=left if left is not None else 0.5)
            except queue.Empty:
                if end is not None and time.monotonic() >= end:
                    return None
                if self._stop.is_set():
                    return None
                continue
            if trig.state == T_PENDING:
                return trig

    # ------------------------------------------------------------------ commands
    def send_pass(self, trigger_id: int) -> CommandResult:
        return self.submit_result(trigger_id, PASS)

    def send_reject(self, trigger_id: int) -> CommandResult:
        return self.submit_result(trigger_id, REJECT)

    def submit_result(self, trigger_id: int, verdict: str) -> CommandResult:
        """Answer one trigger, once. Blocks until the PLC acknowledges or ack_timeout_s passes."""
        v = str(verdict).strip().upper()
        res = CommandResult(trigger_id, v, REFUSED)
        if v not in AM.COMMAND_BITS:
            res.detail = f"unknown verdict {verdict!r} (only PASS or REJECT)"
            self._event("COMMAND_REFUSED", command=v, trigger_id=trigger_id, error=res.detail)
            return res
        try:
            return self._call(lambda: self._submit(trigger_id, v), timeout=self.ack_timeout_s + self.client.timeout * 4 + 10)
        except PLCError as e:                         # service not running / job lost: nothing was sent
            res.detail = f"{type(e).__name__}: {e}"
            self._event("COMMAND_REFUSED", command=v, trigger_id=trigger_id, error=res.detail)
            return res

    def _submit(self, trigger_id: int, v: str) -> CommandResult:
        dev = AM.COMMAND_BITS[v]
        res = CommandResult(trigger_id, v, REFUSED, device=dev)

        def refuse(msg):
            res.status, res.detail = REFUSED, msg
            self._event("COMMAND_REFUSED", device=dev, command=v, trigger_id=trigger_id, error=msg)
            return res

        with self._lock:
            tr = self._trigger
        if tr is None or tr.id != trigger_id:
            return refuse("no such active trigger (cancelled, lost, or already completed)")
        if tr.state != T_PENDING:
            return refuse(f"trigger already {tr.state}: one answer per trigger")
        if self.client.state != CONNECTED:
            return refuse(f"PLC link is {self.client.state}")
        # preconditions, read fresh from the PLC
        try:
            now = self.client.read_many([AM.TRIGGER_BIT, *self._cmd_bits])
        except PLCError as e:
            self._on_link_error(e)
            return refuse(f"precondition read failed: {type(e).__name__}: {e}")
        res.before = dict(now)
        if not now[AM.TRIGGER_BIT]:
            self._observe(now)
            return refuse("M2 is 0: the PLC no longer has an inspection pending")
        if now[AM.PASS_BIT] or now[AM.REJECT_BIT]:
            return refuse(f"a command bit is already ON (M0={now[AM.PASS_BIT]}, M1={now[AM.REJECT_BIT]}): not stacking commands")
        # ---- point of no return: mark answered BEFORE sending, so no path can send it twice
        tr.state = T_ANSWERED
        t0 = time.perf_counter()
        res.write_sent = True
        self._event("COMMAND_SENT", device=dev, command=v, trigger_id=trigger_id)
        try:
            self.client.write_bit(dev, True)
        except PLCError as e:
            res.status = WRITE_FAILED
            res.detail = f"{type(e).__name__}: {e} (outcome unknown; NOT retried)"
            self._unresolved_command = True
            self._event("COMMAND_WRITE_FAILED", device=dev, command=v, trigger_id=trigger_id, error=res.detail)
            self._on_link_error(e)
            return res
        res.write_ms = (time.perf_counter() - t0) * 1000
        self._event("COMMAND_WRITTEN", device=dev, command=v, trigger_id=trigger_id, latency_ms=res.write_ms)
        # ---- acknowledgement: the PLC must consume the command (bit and M2 go back to 0)
        deadline = time.monotonic() + self.ack_timeout_s
        while True:
            try:
                st = self.client.read_many([AM.TRIGGER_BIT, *self._cmd_bits])
            except PLCError as e:
                res.status = ACK_LOST
                res.detail = f"{type(e).__name__}: {e} (command written; ack unknown; NOT retried)"
                self._unresolved_command = True
                self._event("COMMAND_ACK_LOST", device=dev, command=v, trigger_id=trigger_id, error=res.detail)
                self._on_link_error(e)
                return res
            if st[dev]:
                res.cmd_seen_on = True
            held = dev in AM.HELD_UNTIL_DONE
            if not st[AM.TRIGGER_BIT] and (not st[dev] or held):
                res.ack_ms = (time.perf_counter() - t0) * 1000
                res.status = ACKED
                res.cmd_held = bool(st[dev])
                self._unresolved_command = False
                self._event("COMMAND_ACKED", device=dev, command=v, trigger_id=trigger_id,
                            ack=(f"M2 cleared; {dev} held ON by the ladder until the reject cycle ends"
                                 if res.cmd_held else "command bit and M2 cleared by the PLC"),
                            latency_ms=res.ack_ms)
                self._observe(st)
                return res
            if time.monotonic() >= deadline:
                res.status = NOT_ACKED
                res.detail = (f"PLC did not consume the command within {self.ack_timeout_s:g}s "
                              f"(M2={st[AM.TRIGGER_BIT]}, {dev}={st[dev]}); NOT retried")
                self._observe(st)                         # keep the M2 edge tracking in step with what was read
                self._unresolved_command = True
                self._event("COMMAND_NOT_ACKED", device=dev, command=v, trigger_id=trigger_id, error=res.detail)
                return res
            if self.ack_poll_s:
                time.sleep(self.ack_poll_s)
