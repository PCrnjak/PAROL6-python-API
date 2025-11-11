import pytest
from parol6.protocol import wire


def test_encode_move_joint_with_none():
    s = wire.encode_move_joint([1, 2, 3, 4, 5, 6], None, None)
    assert s == "MOVEJOINT|1|2|3|4|5|6|NONE|NONE"


def test_encode_move_joint_with_values():
    s = wire.encode_move_joint([0, -10.5, 20, 30.25, -40, 50], 2.5, 75)
    assert s == "MOVEJOINT|0|-10.5|20|30.25|-40|50|2.5|75"


def test_encode_move_pose():
    s = wire.encode_move_pose([1, 2, 3, 4, 5, 6], 1.0, None)
    assert s == "MOVEPOSE|1|2|3|4|5|6|1.0|NONE"


def test_encode_move_cartesian():
    s = wire.encode_move_cartesian([10, 20, 30, 1, 2, 3], None, 50)
    assert s == "MOVECART|10|20|30|1|2|3|NONE|50"


def test_encode_move_cartesian_rel_trf_variants():
    # Profile/tracking should be upper-cased and None becomes NONE
    s = wire.encode_move_cartesian_rel_trf(
        deltas=[1, 2, 3, 4, 5, 6],
        duration=None,
        speed=50,
        accel=100,
        profile="s_curve",
        tracking="queued",
    )
    assert s == "MOVECARTRELTRF|1|2|3|4|5|6|NONE|50|100|S_CURVE|QUEUED"

    s2 = wire.encode_move_cartesian_rel_trf(
        deltas=[0, 0, 0, 0, 0, 0],
        duration=2.0,
        speed=None,
        accel=None,
        profile=None,
        tracking=None,
    )
    assert s2 == "MOVECARTRELTRF|0|0|0|0|0|0|2.0|NONE|NONE|NONE|NONE"


def test_encode_jog_joint():
    s = wire.encode_jog_joint(3, 80, 0.25, None)
    assert s == "JOG|3|80|0.25|NONE"

    s2 = wire.encode_jog_joint(0, 10, None, 5.5)
    assert s2 == "JOG|0|10|NONE|5.5"


def test_encode_cart_jog():
    s = wire.encode_cart_jog("WRF", "X+", 50, 0.5)
    assert s == "CARTJOG|WRF|X+|50|0.5"


def test_encode_gcode():
    s = wire.encode_gcode("G0 X0 Y0 Z0")
    assert s == "GCODE|G0 X0 Y0 Z0"


def test_encode_gcode_program_inline():
    s = wire.encode_gcode_program_inline(["G21", "G90", "G0 X0 Y0", "G1 X10 F1000"])
    assert s == "GCODE_PROGRAM|INLINE|G21;G90;G0 X0 Y0;G1 X10 F1000"


@pytest.mark.parametrize(
    "resp,prefix,expected",
    [
        ("ANGLES|0,1,2,3,4,5", "ANGLES", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ("IO|1,0,1,0,1", "IO", [1, 0, 1, 0, 1]),
        ("GRIPPER|1,255,150,500,3,1", "GRIPPER", [1, 255, 150, 500, 3, 1]),
        ("SPEEDS|0,0.5,-1,2.5,3,4", "SPEEDS", [0.0, 0.5, -1.0, 2.5, 3.0, 4.0]),
        ("POSE|1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16", "POSE", [float(i) for i in range(1, 17)]),
    ],
)
def test_decode_simple_success(resp, prefix, expected):
    out = wire.decode_simple(resp, prefix)
    assert out == expected


@pytest.mark.parametrize(
    "resp,prefix",
    [
        ("ANGLES|a,b,c", "ANGLES"),
        ("IO|1,2,x", "IO"),
        ("WRONG|1,2,3", "ANGLES"),
        ("", "ANGLES"),
        (None, "ANGLES"),
    ],
)
def test_decode_simple_fail(resp, prefix):
    out = wire.decode_simple(resp, prefix)
    assert out is None


def test_decode_status_success():
    resp = (
        "STATUS|"
        "POSE=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16|"
        "ANGLES=0,10,20,30,40,50|"
        "IO=1,1,0,0,1|"
        "GRIPPER=1,20,150,500,3,1"
    )
    result = wire.decode_status(resp)
    assert result is not None
    assert isinstance(result, dict)
    assert isinstance(result.get("pose"), list)
    assert len(result["pose"]) == 16
    assert result["angles"] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert result["io"] == [1, 1, 0, 0, 1]
    assert result["gripper"] == [1, 20, 150, 500, 3, 1]


def test_decode_status_invalid_returns_none():
    assert wire.decode_status("STATUS|") is None
    assert wire.decode_status("") is None
    assert wire.decode_status("NOTSTATUS|whatever") is None
