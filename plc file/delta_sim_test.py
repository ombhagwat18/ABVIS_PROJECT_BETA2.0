import socket
import time

# ============================================================
# DELTA DVP14SS2 SIMULATOR SETTINGS
# ============================================================

PLC_IP = "127.0.0.1"
PLC_PORT = 10002

# Delta DVP communication station
STATION = 1

# ============================================================
# DVP MODBUS ADDRESS MAP
# ============================================================

# Internal relays
M0_ADDR = 0x0800
M1_ADDR = 0x0801
M2_ADDR = 0x0802

# Outputs
Y0_ADDR = 0x0500
Y1_ADDR = 0x0501


# ============================================================
# LRC CALCULATION
# ============================================================

def calculate_lrc(data):
    """
    Delta Modbus ASCII LRC.
    data = list of binary bytes
    """

    total = sum(data)

    lrc = (-total) & 0xFF

    return lrc


# ============================================================
# BUILD MODBUS ASCII FRAME
# ============================================================

def build_frame(data):
    """
    Convert binary Modbus data into Delta Modbus ASCII frame.

    Example:

    :01050800FF00LRC\r\n
    """

    lrc = calculate_lrc(data)

    frame = ":" + "".join(f"{byte:02X}" for byte in data)

    frame += f"{lrc:02X}"

    frame += "\r\n"

    return frame.encode("ascii")


# ============================================================
# OPEN CONNECTION
# ============================================================

def connect_plc():

    print("Connecting to Delta DVP Simulator...")
    print(f"IP   : {PLC_IP}")
    print(f"PORT : {PLC_PORT}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.settimeout(2)

    sock.connect((PLC_IP, PLC_PORT))

    print("CONNECTED!")

    return sock


# ============================================================
# RECEIVE MODBUS ASCII RESPONSE
# ============================================================

def receive_response(sock):

    data = b""

    while True:

        chunk = sock.recv(1024)

        if not chunk:
            break

        data += chunk

        if b"\r\n" in data:
            break

    return data.decode("ascii", errors="ignore").strip()


# ============================================================
# SEND COMMAND
# ============================================================

def send_command(sock, data):

    frame = build_frame(data)

    print("TX:", frame.decode().strip())

    sock.sendall(frame)

    response = receive_response(sock)

    print("RX:", response)

    return response


# ============================================================
# WRITE SINGLE M BIT
# ============================================================

def write_bit(sock, address, state):

    # Modbus function 05 = Write Single Coil
    #
    # ON  = FF 00
    # OFF = 00 00

    if state:
        value_high = 0xFF
        value_low = 0x00
    else:
        value_high = 0x00
        value_low = 0x00

    data = [
        STATION,
        0x05,

        (address >> 8) & 0xFF,
        address & 0xFF,

        value_high,
        value_low
    ]

    return send_command(sock, data)


# ============================================================
# READ M0, M1, M2
# ============================================================

def read_m_bits(sock):

    # Function 01 = Read Coils
    #
    # Starting address = M0
    # Quantity = 3
    #
    # M0 = bit 0
    # M1 = bit 1
    # M2 = bit 2

    data = [
        STATION,
        0x01,

        (M0_ADDR >> 8) & 0xFF,
        M0_ADDR & 0xFF,

        0x00,
        0x03
    ]

    response = send_command(sock, data)

    if not response:
        return None

    try:

        # Remove :
        response = response[1:]

        # Convert ASCII hexadecimal to bytes
        raw = bytes.fromhex(response)

        # Expected:
        #
        # Station
        # Function
        # Byte count
        # Data
        # LRC

        if len(raw) < 5:
            return None

        station = raw[0]
        function = raw[1]
        byte_count = raw[2]

        if function != 0x01:
            print("Unexpected function code:", hex(function))
            return None

        data_byte = raw[3]

        m0 = bool(data_byte & 0x01)
        m1 = bool(data_byte & 0x02)
        m2 = bool(data_byte & 0x04)

        return m0, m1, m2

    except Exception as e:

        print("Read error:", e)

        return None


# ============================================================
# READ M2 ONLY
# ============================================================

def read_m2(sock):

    result = read_m_bits(sock)

    if result is None:
        return None

    return result[2]


# ============================================================
# MAIN TEST PROGRAM
# ============================================================

def main():

    try:

        sock = connect_plc()

    except Exception as e:

        print()
        print("CONNECTION FAILED")
        print("------------------")
        print(e)
        print()
        print("Check:")
        print("1. COMMGR is running")
        print("2. DVP14_SIM is START")
        print("3. Port is 10002")
        print("4. DVP Simulator is running")
        print()

        return

    print()
    print("====================================")
    print(" DELTA DVP14SS2 PYTHON TEST")
    print("====================================")
    print()
    print("Commands:")
    print()
    print("  g = Send GOOD signal  (M0 ON)")
    print("  r = Send REJECT signal (M1 ON)")
    print("  s = Read M0, M1, M2")
    print("  m = Monitor M2")
    print("  q = Quit")
    print()

    try:

        while True:

            command = input("Command: ").strip().lower()

            # ------------------------------------------------
            # GOOD
            # ------------------------------------------------

            if command == "g":

                print()
                print("Sending GOOD signal...")
                write_bit(sock, M0_ADDR, True)

                time.sleep(0.1)

                result = read_m_bits(sock)

                print("M0/M1/M2 =", result)
                print()

            # ------------------------------------------------
            # REJECT
            # ------------------------------------------------

            elif command == "r":

                print()
                print("Sending DEFECTIVE signal...")
                write_bit(sock, M1_ADDR, True)

                time.sleep(0.1)

                result = read_m_bits(sock)

                print("M0/M1/M2 =", result)
                print()

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            elif command == "s":

                result = read_m_bits(sock)

                print()
                print("M0 =", result[0])
                print("M1 =", result[1])
                print("M2 =", result[2])
                print()

            # ------------------------------------------------
            # MONITOR M2
            # ------------------------------------------------

            elif command == "m":

                print()
                print("Monitoring M2...")
                print("Press CTRL+C to stop.")
                print()

                previous = None

                try:

                    while True:

                        m2 = read_m2(sock)

                        if m2 != previous:

                            print(
                                time.strftime("%H:%M:%S"),
                                "M2 =",
                                "ON" if m2 else "OFF"
                            )

                            previous = m2

                        time.sleep(0.1)

                except KeyboardInterrupt:

                    print()
                    print("M2 monitoring stopped.")
                    print()

            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            elif command == "q":

                break

            else:

                print("Unknown command.")

    finally:

        sock.close()

        print("Disconnected.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()