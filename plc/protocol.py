"""Modbus ASCII framing, as spoken by the Delta DVP simulator on 127.0.0.1:10002.

Pure functions, no I/O, so the framing can be tested without a PLC. Verified against the
running DVP-SS2 simulator (see docs/roadmap/PLC_COMMUNICATION.md):

    request :  ':' + hex(station, function, data...) + hex(LRC) + CR LF
    response:  same shape; function | 0x80 means an exception, followed by one code byte
    LRC     :  (-sum(bytes)) & 0xFF over station..data

Functions used: 0x01 read coils, 0x02 read discrete inputs, 0x03 read holding registers,
0x05 write single coil, 0x06 write single register.
"""
from __future__ import annotations

FC_READ_COILS, FC_READ_INPUTS, FC_READ_REGS = 0x01, 0x02, 0x03
FC_WRITE_COIL, FC_WRITE_REG = 0x05, 0x06
MAX_FRAME = 512                              # bytes on the wire; a runaway line is a protocol error


class PLCError(Exception):
    """Base class: anything that means 'the PLC did not give a trustworthy answer'."""


class PLCConnectionError(PLCError):
    """Not connected, refused, closed, or the transport failed."""


class PLCTimeoutError(PLCError):
    """No complete response within the timeout."""


class PLCProtocolError(PLCError):
    """A response arrived but is malformed: bad framing, LRC, station, function or length."""


class PLCExceptionResponse(PLCError):
    """The PLC answered with a Modbus exception (e.g. 0x02 = illegal data address)."""

    def __init__(self, function: int, code: int):
        super().__init__(f"PLC exception response: function 0x{function:02X}, code 0x{code:02X}")
        self.function, self.code = function, code


def lrc(data: bytes) -> int:
    return (-sum(data)) & 0xFF


def build_frame(data: bytes) -> bytes:
    """data = station, function, payload... -> the ASCII frame including LRC and CRLF."""
    return (":" + data.hex().upper() + f"{lrc(data):02X}" + "\r\n").encode("ascii")


def parse_frame(raw: bytes) -> bytes:
    """ASCII frame -> station..data bytes (LRC stripped). Raises PLCProtocolError on any defect."""
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError as e:
        raise PLCProtocolError(f"non-ASCII response: {raw[:40]!r}") from e
    if not text.startswith(":"):
        raise PLCProtocolError(f"response does not start with ':': {text[:40]!r}")
    body = text[1:]
    if len(body) < 6 or len(body) % 2:
        raise PLCProtocolError(f"bad response length: {text[:60]!r}")
    try:
        data = bytes.fromhex(body)
    except ValueError as e:
        raise PLCProtocolError(f"non-hex response: {text[:60]!r}") from e
    payload, got = data[:-1], data[-1]
    if lrc(payload) != got:
        raise PLCProtocolError(f"LRC mismatch: got 0x{got:02X}, expected 0x{lrc(payload):02X}")
    return payload


def request(station: int, function: int, *fields: int) -> bytes:
    """Build a request frame. Each field is a 16-bit big-endian value (address, count, value)."""
    out = bytearray([station, function])
    for f in fields:
        if not 0 <= f <= 0xFFFF:
            raise ValueError(f"field out of 16-bit range: {f}")
        out += f.to_bytes(2, "big")
    return build_frame(bytes(out))


def check_response(payload: bytes, station: int, function: int) -> bytes:
    """Validate station/function; raise PLCExceptionResponse for an exception reply.
    Returns the bytes after the function code."""
    if len(payload) < 2:
        raise PLCProtocolError(f"response too short: {payload.hex()}")
    if payload[0] != station:
        raise PLCProtocolError(f"response from station {payload[0]}, expected {station}")
    if payload[1] == (function | 0x80):
        if len(payload) != 3:
            raise PLCProtocolError(f"malformed exception response: {payload.hex()}")
        raise PLCExceptionResponse(function, payload[2])
    if payload[1] != function:
        raise PLCProtocolError(f"response function 0x{payload[1]:02X}, expected 0x{function:02X}")
    return payload[2:]


def unpack_bits(body: bytes, count: int) -> list[int]:
    """Read-coils/inputs response body (byte count + packed bits, LSB first) -> list of 0/1."""
    if not body or body[0] != len(body) - 1 or body[0] * 8 < count:
        raise PLCProtocolError(f"bit response length mismatch: {body.hex()} for {count} bits")
    data = body[1:]
    return [(data[i // 8] >> (i % 8)) & 1 for i in range(count)]


def unpack_regs(body: bytes, count: int) -> list[int]:
    if not body or body[0] != len(body) - 1 or body[0] != count * 2:
        raise PLCProtocolError(f"register response length mismatch: {body.hex()} for {count} regs")
    return [int.from_bytes(body[1 + 2 * i: 3 + 2 * i], "big") for i in range(count)]
