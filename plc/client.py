"""PLCClient: the one place the rest of the application talks to a PLC.

    CURRENT:  Python -> TcpTransport -> DVP-SS2 simulator (127.0.0.1:10002, Modbus ASCII) -> ladder
    LATER  :  Python -> a serial/TCP Transport for the wired Delta PLC -> ladder

Only the Transport differs between the two; PLCClient, the address map and everything above them
stay the same. (A serial Transport is NOT implemented -- see SYSTEM_ROADMAP/PLC_COMMUNICATION.md.)

Behaviour that matters for safety:
  * No silent failure. Any problem raises a PLCError subclass and the client state becomes
    FAULT (or DISCONNECTED). A communication failure is never reported as a value.
  * No commands while not CONNECTED. Calls raise immediately; nothing is queued or retried.
  * No hidden retries and no automatic reconnect. reconnect() is an explicit call.
  * After a timeout/protocol error the connection is closed (the stream may be out of step).
  * Writes are refused by policy (address_map.is_writable) before any bytes are sent.
  * Latency uses time.perf_counter(); freshness uses time.monotonic().
"""
from __future__ import annotations

import socket
import threading
import time
from typing import Protocol

from . import address_map as AM
from . import protocol as P
from .protocol import (PLCConnectionError, PLCError, PLCExceptionResponse, PLCProtocolError,  # noqa: F401
                       PLCTimeoutError)

DISCONNECTED, CONNECTED, FAULT = "DISCONNECTED", "CONNECTED", "FAULT"


class PLCWriteNotAllowed(PLCError):
    """The write policy forbids this device. Nothing was sent."""


class Transport(Protocol):
    """What PLCClient needs from a link. Implement this for a serial RS-485 port later."""
    def open(self) -> None: ...
    def close(self) -> None: ...
    def send(self, data: bytes) -> None: ...
    def recv_frame(self, timeout: float) -> bytes: ...
    description: str


class TcpTransport:
    """TCP to the ISPSoft/COMMGR DVP simulator. One connection, CRLF-terminated ASCII frames."""

    def __init__(self, host: str = "127.0.0.1", port: int = 10002, connect_timeout: float = 2.0):
        self.host, self.port, self.connect_timeout = host, port, connect_timeout
        self._sock: socket.socket | None = None
        self.description = f"TCP {host}:{port}"

    def open(self) -> None:
        self.close()
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        except socket.timeout as e:
            raise PLCTimeoutError(f"connect to {self.description} timed out") from e
        except OSError as e:
            raise PLCConnectionError(f"cannot connect to {self.description}: {e}") from e
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s

    def close(self) -> None:
        s, self._sock = self._sock, None
        if s is not None:
            try:
                s.close()
            except OSError:
                pass

    def send(self, data: bytes) -> None:
        if self._sock is None:
            raise PLCConnectionError("transport is not open")
        try:
            self._sock.sendall(data)
        except OSError as e:
            raise PLCConnectionError(f"send failed: {e}") from e

    def recv_frame(self, timeout: float) -> bytes:
        if self._sock is None:
            raise PLCConnectionError("transport is not open")
        deadline = time.monotonic() + timeout
        buf = b""
        while b"\r\n" not in buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise PLCTimeoutError(f"no complete response within {timeout:g}s")
            self._sock.settimeout(left)
            try:
                chunk = self._sock.recv(1024)
            except socket.timeout as e:
                raise PLCTimeoutError(f"no complete response within {timeout:g}s") from e
            except OSError as e:
                raise PLCConnectionError(f"receive failed: {e}") from e
            if not chunk:
                raise PLCConnectionError("connection closed by the PLC")
            buf += chunk
            if len(buf) > P.MAX_FRAME:
                raise PLCProtocolError(f"response longer than {P.MAX_FRAME} bytes without CRLF")
        return buf.split(b"\r\n", 1)[0] + b"\r\n"


class PLCClient:
    def __init__(self, transport: Transport | None = None, station: int = 1, timeout: float = 1.0,
                 target: str = "ISPSoft DVP-SS2 simulator"):
        self.transport = transport or TcpTransport()
        self.station, self.timeout, self.target = station, timeout, target
        self.protocol = "Modbus ASCII"
        self._lock = threading.Lock()
        self.state = DISCONNECTED
        # health
        self.last_ok: float | None = None          # time.monotonic() of the last good reply
        self.last_ok_wall: float | None = None     # time.time(), for display only
        self.last_latency_ms: float | None = None
        self.last_error: str | None = None
        self.n_ok = self.n_err = self.n_exc = self.consecutive_errors = 0

    # ----- connection -------------------------------------------------------------------
    def connect(self) -> float:
        """Open the link and prove the PLC answers (one heartbeat read). Returns the latency in ms.
        On failure the state is FAULT and the error is raised."""
        with self._lock:
            self.state = DISCONNECTED
            try:
                self.transport.open()
            except PLCError as e:
                self._fail(e)
                self.state = FAULT                  # a failed attempt is an error, not a quiet "disconnected"
                raise
            self.state = CONNECTED                # provisional: the heartbeat below must confirm it
        try:
            return self.heartbeat()["latency_ms"]
        except PLCError:
            self.disconnect(fault=True)
            raise

    def disconnect(self, fault: bool = False) -> None:
        """Close the link. Idempotent and never raises. Sends nothing (no output is left commanded:
        this phase never writes)."""
        with self._lock:
            try:
                self.transport.close()
            except Exception:                      # noqa: BLE001 - closing must not fail the caller
                pass
            self.state = FAULT if fault else DISCONNECTED

    shutdown = disconnect

    def reconnect(self) -> float:
        self.disconnect()
        return self.connect()

    def is_connected(self) -> bool:
        return self.state == CONNECTED

    def age_s(self) -> float | None:
        """Seconds since the last good reply (monotonic), or None if there has never been one."""
        return None if self.last_ok is None else time.monotonic() - self.last_ok

    # ----- one transaction --------------------------------------------------------------
    def _fail(self, e: Exception) -> None:
        self.n_err += 1
        self.consecutive_errors += 1
        self.last_error = f"{type(e).__name__}: {e}"

    def _xact(self, function: int, *fields: int) -> bytes:
        with self._lock:
            if self.state != CONNECTED:
                raise PLCConnectionError(f"PLC link is {self.state}"
                                         + (f" ({self.last_error})" if self.last_error else ""))
            frame = P.request(self.station, function, *fields)
            t0 = time.perf_counter()
            try:
                self.transport.send(frame)
                raw = self.transport.recv_frame(self.timeout)
                dt = (time.perf_counter() - t0) * 1000
                body = P.check_response(P.parse_frame(raw), self.station, function)
            except PLCExceptionResponse as e:
                # a valid reply that says "no": the link is fine, the request was not
                self.n_exc += 1
                self.last_error = f"{type(e).__name__}: {e}"
                self._ok((time.perf_counter() - t0) * 1000)
                raise
            except PLCError as e:
                self._fail(e)
                try:
                    self.transport.close()        # the stream may be out of step: do not reuse it
                except Exception:                  # noqa: BLE001
                    pass
                self.state = FAULT
                raise
            self._ok(dt)
            return body

    def _ok(self, latency_ms: float) -> None:
        self.n_ok += 1
        self.consecutive_errors = 0
        self.last_ok, self.last_ok_wall = time.monotonic(), time.time()
        self.last_latency_ms = latency_ms

    # ----- reads ------------------------------------------------------------------------
    def read_bits(self, device: str, count: int = 1) -> list:
        """`count` consecutive bit devices starting at `device` (X via FC02, others via FC01)."""
        kind, _ = AM.parse(device)
        fc = AM.KINDS[kind]["bit_fc"]
        if fc is None:
            raise AM.AddressError(f"{device} is a word device; use read_word")
        if not 1 <= count <= 2000:
            raise ValueError("count must be 1..2000")
        body = self._xact(fc, AM.address_of(device), count)
        return P.unpack_bits(body, count)

    def read_bit(self, device: str) -> bool:
        return bool(self.read_bits(device, 1)[0])

    def read_words(self, device: str, count: int = 1) -> list:
        """Holding registers (FC03): D, and the current values of T and C."""
        kind, _ = AM.parse(device)
        if not AM.KINDS[kind]["words"]:
            raise AM.AddressError(f"{device} has no word value; use read_bit")
        if not 1 <= count <= 125:
            raise ValueError("count must be 1..125")
        return P.unpack_regs(self._xact(P.FC_READ_REGS, AM.address_of(device), count), count)

    def read_word(self, device: str) -> int:
        return self.read_words(device, 1)[0]

    def read_many(self, names, max_gap: int = 16) -> dict:
        """Read several bit devices with as few transactions as possible: devices of one kind that
        lie within `max_gap` of each other share one block read. Returns {NAME: 0/1}."""
        by_kind: dict = {}
        for n in names:
            kind, idx = AM.parse(n)
            by_kind.setdefault(kind, []).append((idx, n.strip().upper()))
        out: dict = {}
        for kind, items in by_kind.items():
            items.sort()
            start = 0
            while start < len(items):
                end = start
                while end + 1 < len(items) and items[end + 1][0] - items[end][0] <= max_gap:
                    end += 1
                lo, hi = items[start][0], items[end][0]
                first = items[start][1]
                bits = self.read_bits(first, hi - lo + 1)
                for idx, nm in items[start:end + 1]:
                    out[nm] = bits[idx - lo]
                start = end + 1
        return out

    def read_words_many(self, names, max_gap: int = 16) -> dict:
        """Word values (T/C current value, D) for several devices, batched like read_many."""
        by_kind: dict = {}
        for n in names:
            kind, idx = AM.parse(n)
            by_kind.setdefault(kind, []).append((idx, n.strip().upper()))
        out: dict = {}
        for kind, items in by_kind.items():
            items.sort()
            start = 0
            while start < len(items):
                end = start
                while end + 1 < len(items) and items[end + 1][0] - items[end][0] <= max_gap:
                    end += 1
                lo, hi = items[start][0], items[end][0]
                vals = self.read_words(items[start][1], hi - lo + 1)
                for idx, nm in items[start:end + 1]:
                    out[nm] = vals[idx - lo]
                start = end + 1
        return out

    def read_inputs(self) -> dict:
        """The mapped X devices."""
        return self.read_many([d.name for d in AM.group("X")])

    def read_outputs(self) -> dict:
        """The mapped Y devices (read-only)."""
        return self.read_many([d.name for d in AM.group("Y")])

    def read_status(self) -> dict:
        """Everything the commissioning view needs, from a handful of block reads."""
        snap = {"inputs": self.read_inputs(), "outputs": self.read_outputs(),
                "internal": self.read_many([d.name for d in AM.group("M") if d.name != AM.RUN_FLAG])}
        snap["timers"] = self.read_words_many([d.name for d in AM.group("T")])
        snap["plc_run"] = self.read_bit(AM.RUN_FLAG)
        snap["health"] = self.health()
        return snap

    # ----- writes (guarded) -------------------------------------------------------------
    def write_bit(self, device: str, state: bool) -> None:
        """Write one bit device. Refused unless address_map.is_writable(device). Never writes X or Y."""
        if not AM.is_writable(device):
            raise PLCWriteNotAllowed(f"writing {device} is not allowed (X/Y are never writable here; "
                                     f"M/D need an explicit entry in address_map.WRITE_ALLOWLIST)")
        self._xact(P.FC_WRITE_COIL, AM.address_of(device), 0xFF00 if state else 0x0000)

    def write_word(self, device: str, value: int) -> None:
        if not AM.is_writable(device):
            raise PLCWriteNotAllowed(f"writing {device} is not allowed")
        if not 0 <= int(value) <= 0xFFFF:
            raise ValueError("word value must be 0..65535")
        self._xact(P.FC_WRITE_REG, AM.address_of(device), int(value))

    # ----- health -----------------------------------------------------------------------
    def heartbeat(self) -> dict:
        """One read of the RUN monitor relay. Proves the PLC answers and reports whether it is in RUN.
        Read-only; raises PLCError on any failure (it never returns 'ok' for a dead link)."""
        t0 = time.perf_counter()
        run = bool(self.read_bits(AM.RUN_FLAG, 1)[0])
        return {"ok": True, "latency_ms": (time.perf_counter() - t0) * 1000, "plc_run": run}

    def health(self) -> dict:
        return {"state": self.state, "target": self.target, "transport": self.transport.description,
                "protocol": self.protocol, "station": self.station, "timeout_s": self.timeout,
                "last_latency_ms": self.last_latency_ms, "age_s": self.age_s(),
                "last_ok_wall": self.last_ok_wall, "last_error": self.last_error,
                "ok": self.n_ok, "errors": self.n_err, "exception_replies": self.n_exc,
                "consecutive_errors": self.consecutive_errors}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()
