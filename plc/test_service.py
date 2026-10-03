"""PLCService tests against a FAKE PLC running FakeLadder (the user-stated contract with scaled timers).

    python -m plc.test_service

FAKE PLC TEST. These prove the service's own logic (edge detection, at-most-once commands, fault
handling). They are not simulator evidence and not physical-PLC evidence: see plc/handshake_test.py
for the real ISPSoft simulator.
"""
from __future__ import annotations

import sys
import threading
import time

from . import address_map as AM
from .client import CONNECTED, DISCONNECTED, FAULT
from .protocol import PLCConnectionError, PLCError
from .service import (ACKED, DEGRADED, NOT_ACKED, REFUSED, T_LOST, WRITE_FAILED, PLCService)
from .test_simulation import FakeLadder, FakePLC, _client, _expect

M = AM.address_of


def rig(**kw):
    fake = FakePLC()
    lad = FakeLadder(fake)
    svc = PLCService(_client(fake, timeout=0.4), poll_s=0.01, status_period_s=0.1, **kw)
    svc.start()
    svc.connect()
    return fake, lad, svc


def close(fake, lad, svc):
    svc.stop(); lad.stop(); fake.close()


def writes(fake, name):
    return [w for w in fake.writes if w[0] == M(name)]


def cmd_writes(fake):
    return [w for w in fake.writes if w[1] == 1]


def test_pass_and_reject():
    fake, lad, svc = rig()
    try:
        assert svc.wait_for_trigger(0.15) is None, "trigger from nothing"
        lad.trigger()
        t = svc.wait_for_trigger(1.0)
        assert t and t.id == 1 and not t.after_reconnect
        time.sleep(0.2)
        assert svc.wait_for_trigger(0.1) is None, "duplicate reads of M2 produced a second trigger"
        r = svc.send_pass(t.id)
        assert r.status == ACKED and r.ok and r.write_ms > 0 and r.ack_ms >= r.write_ms, r
        assert writes(fake, "M0") == [(M("M0"), 1)] and not writes(fake, "M1")
        time.sleep(0.4)
        assert not lad.y0_log, "Y0 fired on PASS"
        assert svc.send_pass(t.id).status == REFUSED, "second answer to the same trigger was not refused"
        assert len(writes(fake, "M0")) == 1, "duplicate PASS reached the PLC"
        lad.trigger()
        t2 = svc.wait_for_trigger(1.0)
        assert t2.id == 2
        r = svc.send_reject(t2.id)
        assert r.status == ACKED and writes(fake, "M1") == [(M("M1"), 1)]
        time.sleep(0.5)
        assert [v for _, v in lad.y0_log] == [1, 0], lad.y0_log
        assert not any(w[0] in (M("Y0"), M("Y1")) for w in fake.writes), "Python wrote a Y output"
        evs = [e.event for e in svc.events()]
        for need in ("CONNECTED", "TRIGGER", "COMMAND_SENT", "COMMAND_WRITTEN", "COMMAND_ACKED", "COMMAND_REFUSED"):
            assert need in evs, (need, evs)
        e = next(e for e in svc.events() if e.event == "COMMAND_ACKED")
        assert e.trigger_id == 1 and e.command == "PASS" and e.device == "M0" and e.latency_ms > 0
    finally:
        close(fake, lad, svc)


def test_refusals_send_nothing():
    fake, lad, svc = rig()
    try:
        assert svc.send_pass(99).status == REFUSED, "answer with no trigger"
        assert svc.submit_result(1, "FORCE_Y0").status == REFUSED
        lad.trigger(); t = svc.wait_for_trigger(1.0)
        fake.bits[M("M1")] = 1; lad.ignore = True                  # a command bit is already ON
        n = len(fake.writes)
        r = svc.send_pass(t.id)
        assert r.status == REFUSED and "already ON" in r.detail and len(fake.writes) == n
        fake.bits[M("M1")] = 0; lad.ignore = False
        fake.bits[M("M2")] = 0; time.sleep(0.15)                   # PLC dropped the trigger by itself
        assert svc.send_pass(t.id).status == REFUSED and len(fake.writes) == n
        assert svc.current_trigger() is None
        assert any(e.event == "TRIGGER_CANCELLED" for e in svc.events())
    finally:
        close(fake, lad, svc)


def test_racing_callers_write_once():
    fake, lad, svc = rig()
    try:
        lad.trigger(); t = svc.wait_for_trigger(1.0)
        out = []
        ts = [threading.Thread(target=lambda v=v: out.append(svc.submit_result(t.id, v))) for v in ("PASS", "REJECT") * 4]
        [x.start() for x in ts]; [x.join() for x in ts]
        assert sum(r.status == ACKED for r in out) == 1 and sum(r.status == REFUSED for r in out) == 7, [r.status for r in out]
        assert len(cmd_writes(fake)) == 1, "concurrent callers produced more than one command"
    finally:
        close(fake, lad, svc)


def test_no_ack_not_retried():
    fake, lad, svc = rig(ack_timeout_s=0.3)
    try:
        lad.ignore = True; lad.trigger(); t = svc.wait_for_trigger(1.0)
        r = svc.send_reject(t.id)
        assert r.status == NOT_ACKED and len(fake.writes) == 1 and "NOT retried" in r.detail, r
        time.sleep(0.3)
        assert len(fake.writes) == 1 and svc.health_check()["unresolved_command"]
        assert svc.send_reject(t.id).status == REFUSED
    finally:
        close(fake, lad, svc)


def test_lost_write_reply_not_retried():
    fake, lad, svc = rig()
    try:
        lad.trigger(); t = svc.wait_for_trigger(1.0)
        fake.mode = "drop_write_reply"                              # the PLC applied it, the answer never arrives
        r = svc.send_reject(t.id)
        assert r.status == WRITE_FAILED and r.write_sent and "NOT retried" in r.detail, r
        assert svc.link_state() == FAULT and len(cmd_writes(fake)) == 1
        fake.mode = "ok"; time.sleep(0.2)
        assert len(cmd_writes(fake)) == 1, "command was resent after the fault"
        assert svc.send_reject(t.id).status == REFUSED, "command accepted while FAULT"
        _expect(PLCConnectionError, lambda: svc.read_bit("M0"), "read during FAULT")
    finally:
        close(fake, lad, svc)


def test_link_loss_with_active_trigger():
    fake, lad, svc = rig()
    try:
        lad.trigger(); t = svc.wait_for_trigger(1.0)
        fake.mode = "close"; time.sleep(0.3)
        assert svc.link_state() == FAULT and t.state == T_LOST and svc.snapshot() is None
        assert "TRIGGER_LOST" in [e.event for e in svc.events()]
        fake.mode = "ok"
        n = len(fake.writes)
        svc.connect()
        t2 = svc.wait_for_trigger(1.0)
        assert t2 and t2.id != t.id and t2.after_reconnect, "stale M2 after reconnect must be flagged"
        assert len(fake.writes) == n, "reconnect replayed a command"
    finally:
        close(fake, lad, svc)


def test_overdue_and_degraded():
    fake, lad, svc = rig(trigger_overdue_s=0.2, stale_after_s=0.25)
    try:
        lad.trigger(); t = svc.wait_for_trigger(1.0); time.sleep(0.5)
        assert t.overdue and "TRIGGER_OVERDUE" in [e.event for e in svc.events()]
        svc._stop.set(); svc._thread.join(1.0)                      # worker gone: replies stop being fresh
        time.sleep(0.4)
        assert svc.client.state == CONNECTED and svc.link_state() == DEGRADED and not svc.is_connected()
    finally:
        lad.stop(); fake.close(); svc.client.disconnect()


def test_faults_end_in_fault():
    for mode in ("wrongstation", "badlrc", "garbage", "silent"):
        fake, lad, svc = rig()
        try:
            fake.mode = mode; time.sleep(0.8)
            assert svc.link_state() == FAULT and svc.client.last_error, mode
            assert any(e.event == "FAULT" for e in svc.events()), mode
            assert svc.snapshot() is None
        finally:
            close(fake, lad, svc)


def test_simulator_test_write_is_narrow():
    from .test_simulation import AM as _AM
    fake, lad, svc = rig()
    try:
        for dev in ("Y0", "Y1", "M5", "T0", "C0"):
            _expect(_AM.AddressError, lambda d=dev: svc.simulator_test_write(d, True), f"sim-test write {dev}")
        assert not fake.writes
        svc.simulator_test_write("M2", True)
        svc.simulator_test_write("X0", True)
        assert fake.writes == [(M("M2"), 1), (M("X0"), 1)], fake.writes
        assert not AM.is_writable("M2") and not AM.is_writable("X0"), "simulator-test opening was not closed again"
        assert not AM.is_writable("Y0") and AM.sim_test_allow("X0") is not None
        svc.client.transport.host = "192.168.1.50"                  # anything but loopback: refused
        _expect(AM.AddressError, lambda: svc.simulator_test_write("M2", False), "non-loopback")
        assert len(fake.writes) == 2 and any(e.event == "SIM_TEST_WRITE" for e in svc.events())
    finally:
        close(fake, lad, svc)


def test_job_traffic_does_not_hide_triggers():
    """Regression (simulator, 2026-10-03): continuous sample() jobs starved polling, the service never saw
    M2 fall, and the next M2 = 1 produced no trigger."""
    fake, lad, svc = rig()
    try:
        lad.trigger(); t1 = svc.wait_for_trigger(1.0)
        stop = threading.Event()

        def flood():
            while not stop.is_set():
                svc.sample(["Y0", "Y1"], ["T0", "T1"])
        th = threading.Thread(target=flood); th.start()
        try:
            assert svc.send_pass(t1.id).ok
            time.sleep(0.15)
            lad.trigger()
            t2 = svc.wait_for_trigger(1.5)
            assert t2 is not None and t2.id != t1.id, "second trigger hidden by job traffic"
        finally:
            stop.set(); th.join()
        lad.ignore = True                                    # NOT_ACKED path: PLC clears M2 only, like the real ladder's REJECT
        r = [None]
        th = threading.Thread(target=lambda: r.__setitem__(0, svc.send_reject(t2.id))); th.start()
        time.sleep(0.1); fake.bits[M("M2")] = 0; th.join()
        assert r[0].status == NOT_ACKED, r[0]
        fake.bits[M("M1")] = 0; lad.ignore = False
        lad.trigger()
        assert svc.wait_for_trigger(1.5) is not None, "trigger after a NOT_ACKED command was missed"
    finally:
        close(fake, lad, svc)


def test_serial_transport_and_reconfigure():
    """SerialTransport framing over pyserial's loop:// (echo) port, and switching the service's link.
    FAKE: no COM port and no PLC are involved; the physical serial link is NOT tested anywhere."""
    from . import protocol as P
    from .client import PLCClient, SerialTransport, serial_ports
    tr = SerialTransport("loop://", 9600, "7E1")
    assert tr.description == "SERIAL loop:// 9600 7E1" and isinstance(serial_ports(), list)
    _expect(ValueError, lambda: SerialTransport("COM1", 9600, "9X9"), "bad serial format")
    tr.open()
    frame = P.request(1, 1, 0x0802, 1)
    tr.send(b"\x00noise" + frame)                           # loop:// echoes it back, with junk in front
    assert tr.recv_frame(0.5) == frame, "serial frame not recovered"
    from .protocol import PLCTimeoutError
    _expect(PLCTimeoutError, lambda: tr.recv_frame(0.1), "serial timeout")
    tr.close()
    _expect(PLCConnectionError, lambda: tr.send(frame), "send on closed serial port")
    dead = PLCClient(SerialTransport("COM_DOES_NOT_EXIST_99"), timeout=0.2)
    _expect(PLCConnectionError, dead.connect, "missing COM port")
    assert dead.state == FAULT and "cannot open" in dead.last_error
    # an echo is not a PLC: the reply to a read is our own request -> protocol error -> FAULT, never "connected"
    echo = PLCClient(SerialTransport("loop://"), timeout=0.2)
    _expect(PLCError, echo.connect, "echo port is not a PLC")
    assert echo.state == FAULT
    # reconfigure: fake A -> fake B, no replay, simulator switches only on loopback TCP
    fake, lad, svc = rig()
    fake2 = FakePLC()
    try:
        lad.trigger(); t = svc.wait_for_trigger(1.0)
        svc.reconfigure(_client(fake2).transport, 1, "fake B")
        assert svc.link_state() == DISCONNECTED and t.state == T_LOST and svc.snapshot() is None
        svc.connect()
        assert svc.link_state() == CONNECTED and not fake2.writes and svc.link_test(5)["replies"] == 5
        svc.reconfigure(SerialTransport("COM_DOES_NOT_EXIST_99"), 1, "serial")
        assert not svc.simulator_mode
        _expect(PLCError, svc.connect, "serial port missing")
        assert svc.link_state() == FAULT
        _expect(AM.AddressError, lambda: svc.simulator_test_write("M2", True), "sim write on a serial link")
    finally:
        fake2.close(); close(fake, lad, svc)


def test_lifecycle():
    s0 = PLCService(_client(FakePLC()), poll_s=0.01)
    _expect(PLCError, s0.connect, "service not started")
    fake, lad, svc = rig()
    svc.stop(); c = fake.count; svc.stop(); time.sleep(0.1)
    assert fake.count == c, "traffic after stop()"
    assert svc.client.state == DISCONNECTED and not fake.writes, "stop() sent a write"
    lad.stop(); fake.close()


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"ok  plc service tests (FAKE PLC + FakeLadder): {len(tests)} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
