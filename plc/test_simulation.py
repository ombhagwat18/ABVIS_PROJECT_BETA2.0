"""Tests for the PLC layer, in two parts that must not be confused:

  python -m plc.test_simulation                    # UNIT tests against an in-process FAKE PLC
  python -m plc.test_simulation --real             # read-only report from the REAL ISPSoft simulator
  python -m plc.test_simulation --real --bench 300 # + read-latency benchmark (read-only)
  python -m plc.test_simulation --real --write-test M<n> --expect Y<k>[,...]
                                                   # write->ladder->read, ONLY on a bit YOU name

The unit tests prove the client's behaviour (framing, timeouts, faults, write policy). They say
nothing about the real ladder. Nothing in --real writes unless --write-test is given, and
--write-test refuses to run unless you name the command bit explicitly: this module never picks one.
"""
from __future__ import annotations

import argparse
import socket
import statistics
import sys
import threading
import time

from . import address_map as AM
from . import protocol as P
from .client import CONNECTED, DISCONNECTED, FAULT, PLCClient, PLCWriteNotAllowed, TcpTransport
from .protocol import (PLCConnectionError, PLCExceptionResponse, PLCProtocolError, PLCTimeoutError)


# ======================================================================================= fake PLC
class FakePLC:
    """A tiny Modbus-ASCII server with a scriptable fault mode and an optional 'ladder' hook.

    Not a model of the real ladder -- only enough PLC to exercise the client."""

    def __init__(self, ladder=None, enforce_station=False):
        self.bits: dict = {}          # coil address -> 0/1 (Y, M, T contacts)
        self.inputs: dict = {}        # discrete-input address -> 0/1 (X)
        self.regs: dict = {}
        self.bits[AM.address_of("M1000")] = 1        # RUN monitor, like the real PLC
        self.mode = "ok"              # ok | silent | garbage | badlrc | close | exception | slow | wrongstation
        self.delay = 0.0
        self.ladder = ladder          # callable(fake) run after every write
        self.enforce_station = enforce_station
        self.count = 0
        self._srv = socket.socket(); self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0)); self._srv.listen(8)
        self.port = self._srv.getsockname()[1]
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self):
        self._stop = True
        try:
            self._srv.close()
        except OSError:
            pass

    def _accept(self):
        while not self._stop:
            try:
                c, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c):
        buf = b""
        try:
            while True:
                chunk = c.recv(1024)
                if not chunk:
                    return
                buf += chunk
                while b"\r\n" in buf:
                    line, buf = buf.split(b"\r\n", 1)
                    self.count += 1
                    reply = self._handle(line + b"\r\n")
                    if reply == "CLOSE":
                        c.close(); return
                    if reply is not None:
                        if self.delay:
                            time.sleep(self.delay)
                        c.sendall(reply)
        except OSError:
            return

    def _handle(self, raw):
        if self.mode == "silent":
            return None
        if self.mode == "close":
            return "CLOSE"
        if self.mode == "garbage":
            return b"hello this is not modbus\r\n"
        p = P.parse_frame(raw)
        st, fc, body = p[0], p[1], p[2:]
        if self.enforce_station and st != 1:
            return None
        if self.mode == "wrongstation":
            st = 9
        if self.mode == "exception":
            return P.build_frame(bytes([st, fc | 0x80, 0x02]))
        a = int.from_bytes(body[0:2], "big")
        n = int.from_bytes(body[2:4], "big") if len(body) >= 4 else 0
        if fc in (1, 2):
            mem = self.inputs if fc == 2 else self.bits
            if fc == 2 and not (0x0400 <= a < 0x0500) or fc == 1 and (0x0400 <= a < 0x0500):
                return P.build_frame(bytes([st, fc | 0x80, 0x02]))
            by = bytearray((n + 7) // 8)
            for i in range(n):
                if mem.get(a + i, 0):
                    by[i // 8] |= 1 << (i % 8)
            out = bytes([st, fc, len(by)]) + bytes(by)
        elif fc == 3:
            out = bytes([st, fc, n * 2]) + b"".join(self.regs.get(a + i, 0).to_bytes(2, "big") for i in range(n))
        elif fc == 5:
            self.bits[a] = 1 if body[2] == 0xFF else 0
            out = bytes([st, fc]) + body[:4]
            if self.ladder:
                self.ladder(self)
        elif fc == 6:
            self.regs[a] = int.from_bytes(body[2:4], "big")
            out = bytes([st, fc]) + body[:4]
        else:
            return P.build_frame(bytes([st, fc | 0x80, 0x01]))
        frame = P.build_frame(out)
        if self.mode == "badlrc":
            frame = frame[:-4] + (b"00" if frame[-4:-2] != b"00" else b"11") + b"\r\n"
        return frame


def _expect(exc, fn, what):
    try:
        fn()
    except exc:
        return
    except Exception as e:                        # noqa: BLE001
        raise AssertionError(f"{what}: expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"{what}: expected {exc.__name__}, nothing raised")


def _client(fake, timeout=0.5):
    return PLCClient(TcpTransport("127.0.0.1", fake.port, connect_timeout=1.0), station=1, timeout=timeout,
                     target="fake PLC")


# ================================================================================ write->ladder->read
def write_readback_test(plc, cmd, expects, settle=1.0, poll=0.01):
    """Write `cmd` ON, wait for every `expect` device to change, then restore `cmd` and the state.

    Proves the full path: Python write -> PLC -> ladder scan -> device change -> Python read.
    Returns a dict with before/after values and the write->change latency. ALWAYS restores `cmd`
    to its original value, even on failure. `cmd` must already be on the write allow-list."""
    expects = [e.strip().upper() for e in expects]
    cmd = cmd.strip().upper()
    before = plc.read_many(expects + [cmd])
    original = bool(before[cmd])
    result = {"cmd": cmd, "before": dict(before), "changed": {}, "restored": None}
    try:
        t0 = time.perf_counter()
        plc.write_bit(cmd, not original)
        t_write = (time.perf_counter() - t0) * 1000
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline and len(result["changed"]) < len(expects):
            now = plc.read_many(expects)
            for e in expects:
                if e not in result["changed"] and now[e] != before[e]:
                    result["changed"][e] = (before[e], now[e], (time.perf_counter() - t0) * 1000)
            time.sleep(poll)
        result["write_ms"] = t_write
        result["after"] = plc.read_many(expects + [cmd])
    finally:
        plc.write_bit(cmd, original)                      # restore the command bit no matter what
        result["restored"] = plc.read_bit(cmd) == original
    result["ok"] = len(result["changed"]) == len(expects) and result["restored"]
    return result


# ====================================================================================== unit tests
def unit_tests():
    # ---- protocol: frames captured from the real simulator on 2026-10-02
    assert P.request(1, 1, 0x0800, 16) == b":010108000010E6\r\n"
    assert P.request(1, 1, 0x0500, 8) == b":010105000008F1\r\n"
    assert P.unpack_bits(P.check_response(P.parse_frame(b":0101020200FA\r\n"), 1, 1), 16)[:3] == [0, 1, 0]
    _expect(PLCExceptionResponse, lambda: P.check_response(P.parse_frame(b":0181027C\r\n"), 1, 1), "exception frame")
    try:
        P.check_response(P.parse_frame(b":0181027C\r\n"), 1, 1)
    except PLCExceptionResponse as e:
        assert e.code == 2 and e.function == 1
    for bad in (b"0101020200FA\r\n", b":0101020200FB\r\n", b":01010202ZZFA\r\n", b":0\r\n", b"\xff\xfe\r\n"):
        _expect(PLCProtocolError, lambda b=bad: P.parse_frame(b), f"bad frame {bad!r}")
    _expect(PLCProtocolError, lambda: P.check_response(P.parse_frame(P.build_frame(bytes([2, 1, 1, 0]))), 1, 1), "wrong station")
    _expect(PLCProtocolError, lambda: P.unpack_bits(bytes([1, 0]), 16), "short bit body")
    _expect(PLCProtocolError, lambda: P.unpack_regs(bytes([2, 0, 0]), 2), "short register body")
    _expect(ValueError, lambda: P.request(1, 3, 70000), "field out of range")

    # ---- addressing: Delta numbering (X/Y octal; M/T/C/D decimal)
    assert AM.address_of("X0") == 0x0400 and AM.address_of("X7") == 0x0407 and AM.address_of("X10") == 0x0408
    assert AM.address_of("Y2") == 0x0502 and AM.address_of("Y10") == 0x0508
    assert AM.address_of("M10") == 0x080A and AM.address_of("M1000") == 0x0BE8
    assert AM.address_of("T3") == 0x0603 and AM.address_of("D100") == 0x1064
    for bad in ("X8", "Y9", "Q1", "M", "10", "", "M-1", "X 1"):
        _expect(AM.AddressError, lambda b=bad: AM.parse(b), f"invalid device {bad!r}")
    assert AM.PLC_MAP["M0"].meaning == AM.UNKNOWN and AM.PLC_MAP["Y2"].meaning == AM.UNKNOWN
    assert all(d.meaning == AM.UNKNOWN for d in AM.PLC_MAP.values() if d.name != AM.RUN_FLAG), "no guessed meanings"
    assert not any(d.writable for d in AM.PLC_MAP.values()), "nothing writable by default"

    # ---- connect / read / write against the fake
    fake = FakePLC()
    fake.bits[AM.address_of("M1")] = 1
    fake.inputs[AM.address_of("X2")] = 1
    fake.regs[AM.address_of("T3")] = 77
    try:
        plc = _client(fake)
        assert plc.state == DISCONNECTED and not plc.is_connected()
        _expect(PLCConnectionError, lambda: plc.read_bit("M0"), "read before connect")
        assert fake.count == 0, "a command was sent while disconnected"
        lat = plc.connect()                                               # connect success + heartbeat
        assert plc.is_connected() and plc.state == CONNECTED and lat >= 0
        assert plc.heartbeat()["plc_run"] is True
        assert plc.read_bit("M1") is True and plc.read_bit("M0") is False
        assert plc.read_bits("M0", 3) == [0, 1, 0]
        assert plc.read_bit("X2") is True and plc.read_bit("X1") is False          # FC02 path
        assert plc.read_word("T3") == 77
        st = plc.read_status()
        assert st["inputs"]["X2"] == 1 and st["internal"]["M1"] == 1 and st["plc_run"] and st["timers"]["T3"] == 77
        assert st["health"]["state"] == CONNECTED and st["health"]["errors"] == 0

        # read_many groups neighbours into one block read
        c0 = fake.count
        plc.read_many(["M0", "M1", "M2", "M10", "M11", "M12"])
        assert fake.count - c0 == 1, "neighbouring devices should share one transaction"
        fake.regs[AM.address_of("T0")] = 5
        c0 = fake.count
        assert plc.read_words_many(["T0", "T1", "T3", "T4", "T5"]) == {"T0": 5, "T1": 0, "T3": 77, "T4": 0, "T5": 0}
        assert fake.count - c0 == 1, "timers should be one block read"
        c0 = fake.count
        plc.read_many(["M0", "M1000"])
        assert fake.count - c0 == 2, "distant devices should not be merged into a huge read"

        # ---- write policy: nothing is sent when refused
        c0 = fake.count
        for dev in ("Y0", "Y2", "X0", "M5", "D1"):
            _expect(PLCWriteNotAllowed, lambda d=dev: plc.write_bit(d, True), f"write {dev} refused")
        _expect(PLCWriteNotAllowed, lambda: plc.write_word("D1", 5), "write_word refused")
        assert fake.count == c0, "a refused write still reached the PLC"
        AM.WRITE_ALLOWLIST = frozenset({"M5", "Y2"})                      # explicit, by someone who read the ladder
        try:
            plc.write_bit("M5", True)
            assert fake.bits[AM.address_of("M5")] == 1 and plc.read_bit("M5") is True
            _expect(PLCWriteNotAllowed, lambda: plc.write_bit("Y2", True), "Y never writable even if listed")
            assert AM.is_writable("M5") and not AM.is_writable("M6")
        finally:
            AM.WRITE_ALLOWLIST = frozenset()

        # ---- exception reply: link stays up
        fake.mode = "exception"
        _expect(PLCExceptionResponse, lambda: plc.read_bit("M0"), "exception reply")
        assert plc.state == CONNECTED and plc.health()["exception_replies"] == 1
        fake.mode = "ok"
        assert plc.read_bit("M1") is True

        # ---- timeout -> FAULT, link closed, no further commands, explicit reconnect recovers
        fake.mode = "silent"
        t0 = time.monotonic()
        _expect(PLCTimeoutError, lambda: plc.read_bit("M0"), "silent PLC")
        assert 0.4 <= time.monotonic() - t0 < 1.5, "timeout not honoured"
        assert plc.state == FAULT and not plc.is_connected() and "Timeout" in (plc.last_error or "")
        c0 = fake.count
        _expect(PLCConnectionError, lambda: plc.read_bit("M0"), "command while FAULT")
        assert fake.count == c0, "a command was sent while the link was FAULT"
        fake.mode = "ok"
        plc.reconnect()
        assert plc.is_connected() and plc.read_bit("M1") is True and plc.consecutive_errors == 0

        # ---- protocol faults: garbage, bad LRC, wrong station, closed by peer
        for mode, exc in (("garbage", PLCProtocolError), ("badlrc", PLCProtocolError),
                          ("wrongstation", PLCProtocolError), ("close", PLCConnectionError)):
            plc.reconnect() if not plc.is_connected() else None
            fake.mode = mode
            _expect(exc, lambda: plc.read_bit("M0"), f"fake mode {mode}")
            assert plc.state == FAULT, mode
            fake.mode = "ok"
        plc.reconnect()

        # ---- staleness: age grows, never reset by a failure
        a0 = plc.age_s(); time.sleep(0.05); assert plc.age_s() > a0
        fake.mode = "silent"
        _expect(PLCTimeoutError, lambda: plc.read_bit("M0"), "timeout again")
        assert plc.age_s() > 0.4, "age was reset by a failed read"
        fake.mode = "ok"

        # ---- safe shutdown: idempotent, never raises, sends nothing
        c0 = fake.count
        plc.disconnect(); plc.disconnect(); plc.shutdown()
        assert plc.state == DISCONNECTED and fake.count == c0
        _expect(PLCConnectionError, lambda: plc.read_bit("M0"), "read after disconnect")

        # ---- connect failure (nothing listening)
        s = socket.socket(); s.bind(("127.0.0.1", 0)); dead = s.getsockname()[1]; s.close()
        bad = PLCClient(TcpTransport("127.0.0.1", dead, connect_timeout=0.5), timeout=0.3)
        # Windows retries a refused loopback connect, so a short timeout surfaces as a timeout rather
        # than "refused": both are a PLCError, both must leave the link FAULT and unusable.
        _expect((PLCConnectionError, PLCTimeoutError), bad.connect, "nothing listening")
        assert bad.state == FAULT and bad.n_err == 1 and not bad.is_connected()

        # ---- connect to something that connects but never answers the heartbeat
        fake.mode = "silent"
        c = _client(fake, timeout=0.3)
        _expect(PLCTimeoutError, c.connect, "connect + silent heartbeat")
        assert c.state == FAULT
        fake.mode = "ok"

        # ---- thread safety: concurrent readers never interleave frames
        plc = _client(fake); plc.connect()
        errs = []

        def worker():
            try:
                for _ in range(80):
                    assert plc.read_bits("M0", 3) == [0, 1, 0]
            except Exception as e:                    # noqa: BLE001
                errs.append(e)
        ts = [threading.Thread(target=worker) for _ in range(6)]
        [t.start() for t in ts]; [t.join() for t in ts]
        assert not errs, errs[:2]
        plc.disconnect()
    finally:
        fake.close()

    # ---- write->ladder->read machinery, against a fake ladder (M100 drives Y1 and Y2)
    def ladder(f):
        f.bits[AM.address_of("Y1")] = f.bits.get(AM.address_of("M100"), 0)
        f.bits[AM.address_of("Y2")] = f.bits.get(AM.address_of("M100"), 0)
    fake = FakePLC(ladder=ladder)
    try:
        plc = _client(fake); plc.connect()
        AM.WRITE_ALLOWLIST = frozenset({"M100", "M101"})
        try:
            r = write_readback_test(plc, "M100", ["Y1", "Y2"])
            assert r["ok"] and set(r["changed"]) == {"Y1", "Y2"} and r["restored"], r
            assert plc.read_bit("M100") is False and plc.read_bit("Y1") is False, "state not restored"
            # a bit whose ladder does nothing: test reports failure, but still restores the command bit
            r = write_readback_test(plc, "M101", ["Y1"], settle=0.15)
            assert not r["ok"] and r["changed"] == {} and r["restored"], r   # no reaction -> FAIL, bit still restored
            assert plc.read_bit("M101") is False
        finally:
            AM.WRITE_ALLOWLIST = frozenset()
        _expect(PLCWriteNotAllowed, lambda: write_readback_test(plc, "M100", ["Y1"]), "write test needs allow-list")
        plc.disconnect()
    finally:
        fake.close()
    print("ok  plc unit tests (fake PLC): protocol frames captured from the real simulator, octal/decimal addressing, "
          "no guessed meanings, read/write, write policy (refused writes send nothing), exception reply, timeout, "
          "garbage/bad LRC/wrong station/peer close, FAULT blocks commands, reconnect, staleness, safe shutdown, "
          "connect failure, threads, write->ladder->read machinery")


# ===================================================================================== real simulator
def _client_real(a):
    return PLCClient(TcpTransport(a.host, a.port), station=a.station, timeout=a.timeout)


def real_report(a) -> int:
    plc = _client_real(a)
    print("PLC Communication Test")
    print("----------------------")
    print("Target    : ISPSoft DVP-SS2 simulator (as configured in COMMGR 'Simulaiton SE')")
    print(f"Host      : {a.host}")
    print(f"Port      : {a.port}")
    print(f"Station   : {a.station}")
    print("Protocol  : Modbus ASCII over TCP (':' hex LRC CRLF)")
    try:
        lat = plc.connect()
    except Exception as e:                            # noqa: BLE001
        print(f"Connection: FAIL  ({type(e).__name__}: {e})")
        return 1
    print(f"Connection: PASS  (first heartbeat {lat:.1f} ms)")
    try:
        st = plc.read_status()
        h = st["health"]
        print(f"PLC mode  : {'RUN' if st['plc_run'] else 'NOT RUN'}   (M1000 RUN monitor)")
        print(f"Last reply: {h['age_s']:.2f}s ago, {h['last_latency_ms']:.1f} ms, errors {h['errors']}")
        print()
        for title, key, order in (("INPUTS", "inputs", AM.group("X")), ("MEMORY", "internal", AM.group("M")),
                                  ("OUTPUTS", "outputs", AM.group("Y"))):
            print(title)
            for d in order:
                if d.name in st[key]:
                    tag = "" if d.in_ladder_list else "   (not in the ladder-device list)"
                    print(f"  {d.name:<5}= {st[key][d.name]}   @0x{d.address:04X}  meaning={d.meaning}{tag}")
        print("TIMERS (current value, FC03)")
        for d in AM.group("T"):
            print(f"  {d.name:<5}= {st['timers'][d.name]}   @0x{d.address:04X}  meaning={d.meaning}")
    except Exception as e:                            # noqa: BLE001
        print(f"READ FAIL: {type(e).__name__}: {e}")
        return 1
    finally:
        plc.disconnect()
    return 0


def _stats(vals):
    v = sorted(vals)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return (f"mean {statistics.mean(v):6.2f}  median {statistics.median(v):6.2f}  p95 {q(.95):6.2f}  "
            f"p99 {q(.99):6.2f}  min {v[0]:6.2f}  max {v[-1]:6.2f} ms")


def bench(a) -> int:
    print(f"\nISPSoft SIMULATOR communication benchmark (read-only, N={a.bench}) -- NOT real-PLC performance")
    plc = _client_real(a); plc.connect()
    cases = [("single bit read  (FC01)  M1000", lambda: plc.read_bit("M1000")),
             ("block read 16 bits (FC01) M0..M15", lambda: plc.read_bits("M0", 16)),
             ("input read      (FC02)  X0..X7", lambda: plc.read_bits("X0", 8)),
             ("word read      (FC03)  T0..T5", lambda: plc.read_words("T0", 6)),
             ("read_status()  (whole snapshot)", plc.read_status)]
    fails = 0
    try:
        for name, fn in cases:
            lat, f = [], 0
            for _ in range(a.bench):
                t = time.perf_counter()
                try:
                    fn()
                    lat.append((time.perf_counter() - t) * 1000)
                except Exception:                      # noqa: BLE001
                    f += 1
                    if not plc.is_connected():
                        try:
                            plc.reconnect()
                        except Exception:              # noqa: BLE001
                            break
            fails += f
            print(f"  {name:36s} {_stats(lat) if lat else 'no data'}   failures {f}")
    finally:
        plc.disconnect()
    # reconnect reliability: connect + first heartbeat + disconnect, repeated
    n_cycles = min(a.bench, 50)
    cyc, cf = [], 0
    for _ in range(n_cycles):
        c = _client_real(a)
        t = time.perf_counter()
        try:
            c.connect()
            cyc.append((time.perf_counter() - t) * 1000)
        except Exception:                              # noqa: BLE001
            cf += 1
        finally:
            c.disconnect()
    fails += cf
    print(f"  {'connect+heartbeat+disconnect':36s} {_stats(cyc) if cyc else 'no data'}   failures {cf}  (N={n_cycles})")
    # failure behaviour: a port nothing listens on must end in FAULT, quickly and loudly
    dead = PLCClient(TcpTransport(a.host, a.port + 1, connect_timeout=3.0), station=a.station, timeout=a.timeout)
    t = time.perf_counter()
    try:
        dead.connect()
        outcome = "UNEXPECTEDLY CONNECTED"
    except Exception as e:                             # noqa: BLE001
        outcome = f"{type(e).__name__}"
    print(f"  wrong port {a.port + 1}: {outcome} after {(time.perf_counter() - t) * 1000:.0f} ms, "
          f"state={dead.state}, is_connected={dead.is_connected()}")
    print(f"  total failures/timeouts: {fails}")
    return 0 if fails == 0 else 1


def real_write_test(a) -> int:
    cmd = a.write_test.strip().upper()
    expects = [e for e in (a.expect or "").split(",") if e.strip()]
    if not expects:
        print("--write-test needs --expect DEV[,DEV...]: the devices you expect the ladder to change.")
        return 2
    AM.WRITE_ALLOWLIST = frozenset({cmd})              # explicit, named on the command line, this process only
    plc = _client_real(a); plc.connect()
    try:
        print(f"\nWRITE -> LADDER -> READ on the ISPSoft SIMULATOR: command bit {cmd}, expecting {expects} to change")
        r = write_readback_test(plc, cmd, expects, settle=a.settle)
        print(f"  before  : {r['before']}")
        print(f"  write   : {cmd} {'ON' if not r['before'][cmd] else 'OFF'} accepted in {r['write_ms']:.1f} ms")
        for e, (b, af, ms) in r["changed"].items():
            print(f"  changed : {e} {b} -> {af}   {ms:.1f} ms after the write started")
        for e in expects:
            if e not in r["changed"]:
                print(f"  NO CHANGE: {e} did not change within {a.settle:g}s")
        print(f"  restored: {cmd} back to original: {r['restored']}")
        print("  RESULT  :", "PASS" if r["ok"] else "FAIL")
        return 0 if r["ok"] else 1
    finally:
        AM.WRITE_ALLOWLIST = frozenset()
        plc.disconnect()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="talk to the real ISPSoft simulator (read-only)")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=10002)
    ap.add_argument("--station", type=int, default=1); ap.add_argument("--timeout", type=float, default=1.0)
    ap.add_argument("--bench", type=int, default=0, metavar="N", help="with --real: read-latency benchmark")
    ap.add_argument("--write-test", metavar="Mn", help="with --real: write->ladder->read on THIS bit only")
    ap.add_argument("--expect", metavar="DEV[,DEV]", help="with --write-test: devices that should change")
    ap.add_argument("--settle", type=float, default=1.0)
    a = ap.parse_args(argv)
    if not a.real:
        unit_tests()
        return 0
    rc = real_report(a)
    if rc == 0 and a.bench:
        rc = bench(a)
    if rc == 0 and a.write_test:
        rc = real_write_test(a)
    return rc


if __name__ == "__main__":
    sys.exit(main())
