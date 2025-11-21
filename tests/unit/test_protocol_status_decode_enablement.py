import pytest

from parol6.protocol import wire


@pytest.mark.unit
def test_decode_status_with_enablement_arrays():
    # Minimal valid STATUS with ANGLES present to satisfy parser
    joint_en = ",".join(["1", "0"] * 6)
    cart_wrf = ",".join(["1"] * 12)
    cart_trf = ",".join(["0"] * 12)
    status = (
        "STATUS|ANGLES=0,0,0,0,0,0"
        f"|JOINT_EN={joint_en}"
        f"|CART_EN_WRF={cart_wrf}"
        f"|CART_EN_TRF={cart_trf}"
    )

    decoded = wire.decode_status(status)
    assert decoded is not None
    assert isinstance(decoded.get("joint_en"), list)
    assert isinstance(decoded.get("cart_en_wrf"), list)
    assert isinstance(decoded.get("cart_en_trf"), list)
    je = decoded.get("joint_en")
    wrf = decoded.get("cart_en_wrf")
    trf = decoded.get("cart_en_trf")
    assert isinstance(je, list) and len(je) == 12
    assert isinstance(wrf, list) and len(wrf) == 12
    assert isinstance(trf, list) and len(trf) == 12

    # Spot-check values
    assert je[0] == 1 and je[1] == 0
    assert all(v == 1 for v in wrf)  # all ones
    assert all(v == 0 for v in trf)  # all zeros
