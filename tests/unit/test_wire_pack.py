import pytest
import numpy as np

from parol6.protocol import wire
from parol6.protocol.wire import CommandCode


def test_pack_tx_frame_structure_and_command_byte():
    position_out = [1, 2, 3, 4, 5, 6]
    speed_out = [10, 20, 30, 40, 50, 60]
    affected_joint_out = [1, 0, 0, 0, 0, 0, 0, 1]  # MSB..LSB
    inout_out = [0, 1, 0, 1, 0, 1, 0, 1]           # MSB..LSB
    timeout_out = 7
    gripper_data_out = [123, 45, 67, 3, 0, 5]  # pos, spd, cur, cmd, mode, id

    frame = wire.pack_tx_frame(
        position_out,
        speed_out,
        CommandCode.MOVE,
        affected_joint_out,
        inout_out,
        timeout_out,
        gripper_data_out,
    )

    # Structure: 3 start + 1 len + 52 payload + 2 end = 58 bytes
    assert isinstance(frame, (bytes, bytearray))
    assert len(frame) == 58
    assert frame[:3] == b"\xff\xff\xff"
    assert frame[3] == 52
    assert frame[-2:] == b"\x01\x02"

    # Command byte position = 3 (start) + 1 (len) + 18 (pos) + 18 (spd) = 40
    assert frame[40] == int(CommandCode.MOVE)


def test_unpack_rx_frame_happy_path_and_signs():
    # Build a 52-byte payload per firmware layout
    payload = bytearray(52)

    # Positions: 6 * 3 bytes (signed 24-bit two's complement)
    positions = [-1, 0, 1, 1000, -1000, 123456]
    off = 0
    for v in positions:
        b0, b1, b2 = wire.split_to_3_bytes(v)
        payload[off : off + 3] = bytes([b0, b1, b2])
        off += 3

    # Speeds: 6 * 3 bytes
    speeds = [0, 5, -5, 9999, -9999, 654321]
    for v in speeds:
        b0, b1, b2 = wire.split_to_3_bytes(v)
        payload[off : off + 3] = bytes([b0, b1, b2])
        off += 3

    # Homed / IO / errors (bytes 36..39)
    payload[36] = 0xFF  # all homed
    payload[37] = 0xAA  # 10101010 (MSB..LSB)
    payload[38] = 0x00  # no temp errors
    payload[39] = 0x00  # no position errors

    # Timing (bytes 40..41 => low 2 bytes in 24-bit value)
    payload[40] = 0x12
    payload[41] = 0x34

    # Bytes 42..43 unspecified/legacy (ignored by unpacker)

    # Device + gripper (bytes 44..51)
    payload[44] = 7  # device id
    payload[45] = 0x01  # pos hi
    payload[46] = 0x02  # pos lo => 0x0102 = 258
    payload[47] = 0x00  # spd hi
    payload[48] = 0x64  # spd lo => 100
    payload[49] = 0x00  # cur hi
    payload[50] = 0x0A  # cur lo => 10

    # Status byte: bits[2]=1, bits[3]=1 => obj = (1<<1)|1 = 3
    payload[51] = 0b00001100

    # Decode via zero-allocation into-variant
    pos = np.zeros(6, dtype=np.int32)
    spd = np.zeros(6, dtype=np.int32)
    homed = np.zeros(8, dtype=np.uint8)
    io_bits = np.zeros(8, dtype=np.uint8)
    temp = np.zeros(8, dtype=np.uint8)
    poserr = np.zeros(8, dtype=np.uint8)
    timing = np.zeros(1, dtype=np.int32)
    grip = np.zeros(6, dtype=np.int32)

    ok = wire.unpack_rx_frame_into(
        memoryview(payload),
        pos_out=pos,
        spd_out=spd,
        homed_out=homed,
        io_out=io_bits,
        temp_out=temp,
        poserr_out=poserr,
        timing_out=timing,
        grip_out=grip,
    )
    assert ok

    # Validate positions and speeds round trip
    assert pos.tolist() == positions
    assert spd.tolist() == speeds

    # Validate bitfields
    assert homed.tolist() == [1] * 8
    assert io_bits.tolist() == [1, 0, 1, 0, 1, 0, 1, 0]  # MSB..LSB of 0xAA

    # Errors empty
    assert temp.tolist() == [0] * 8
    assert poserr.tolist() == [0] * 8

    # Timing (0x1234 == 4660)
    assert int(timing[0]) == 0x1234

    # Gripper data
    assert grip.tolist() == [7, 258, 100, 10, 0b00001100, 3]
