"""PLC communication layer (Delta DVP via Modbus ASCII). See docs/roadmap/PLC_COMMUNICATION.md.

    from plc import PLCClient
    svc = PLCService(PLCClient())            # the ISPSoft DVP-SS2 simulator on 127.0.0.1:10002
    svc.start(); svc.connect()
    trig = svc.wait_for_trigger()            # M2 rising edge from the PLC
    svc.send_pass(trig.id)                   # or send_reject(trig.id): M0 / M1, once, acknowledged
"""
from . import address_map
from .client import (CONNECTED, DISCONNECTED, FAULT, SERIAL_FORMATS, PLCClient, PLCWriteNotAllowed, SerialTransport,
                     TcpTransport, Transport, serial_ports)
from .service import (ACKED, DEGRADED, NOT_ACKED, PASS, REFUSED, REJECT, WRITE_FAILED, CommandResult, PLCEvent,
                      PLCLink, PLCService, Trigger)
from .protocol import (PLCConnectionError, PLCError, PLCExceptionResponse, PLCProtocolError,
                       PLCTimeoutError)

__all__ = ["PLCClient", "TcpTransport", "SerialTransport", "serial_ports", "SERIAL_FORMATS", "Transport", "address_map", "PLCError", "PLCConnectionError",
           "PLCTimeoutError", "PLCProtocolError", "PLCExceptionResponse", "PLCWriteNotAllowed",
           "CONNECTED", "DISCONNECTED", "FAULT", "DEGRADED", "PLCService", "PLCLink", "Trigger",
           "CommandResult", "PLCEvent", "PASS", "REJECT", "ACKED", "REFUSED", "WRITE_FAILED", "NOT_ACKED"]
