"""PLC communication layer (Delta DVP via Modbus ASCII). See SYSTEM_ROADMAP/PLC_COMMUNICATION.md.

    from plc import PLCClient
    with PLCClient() as plc:                 # the ISPSoft DVP-SS2 simulator on 127.0.0.1:10002
        print(plc.read_status())
"""
from . import address_map
from .client import (CONNECTED, DISCONNECTED, FAULT, PLCClient, PLCWriteNotAllowed, TcpTransport,
                     Transport)
from .protocol import (PLCConnectionError, PLCError, PLCExceptionResponse, PLCProtocolError,
                       PLCTimeoutError)

__all__ = ["PLCClient", "TcpTransport", "Transport", "address_map", "PLCError", "PLCConnectionError",
           "PLCTimeoutError", "PLCProtocolError", "PLCExceptionResponse", "PLCWriteNotAllowed",
           "CONNECTED", "DISCONNECTED", "FAULT"]
