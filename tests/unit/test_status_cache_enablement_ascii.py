import numpy as np
import pytest

from parol6.server.status_cache import StatusCache
from parol6.server.state import ControllerState


@pytest.mark.unit
def test_status_cache_includes_enablement_fields(monkeypatch):
    cache = StatusCache()

    # Patch heavy dependencies to be deterministic and fast
    monkeypatch.setattr(
        "parol6.server.status_cache.PAROL6_ROBOT",
        type(
            "_Dummy",
            (),
            {
                "ops": type(
                    "_Ops",
                    (),
                    {
                        "steps_to_deg": staticmethod(lambda steps: [0.0] * 6),
                        "steps_to_rad": staticmethod(lambda steps: [0.0] * 6),
                    },
                )(),
                "robot": type(
                    "_Robot", (), {"qlim": np.vstack([[-3.14] * 6, [3.14] * 6])}
                )(),
            },
        ),
        raising=True,
    )

    # Short-circuit fkine to avoid spatialmath calls
    monkeypatch.setattr(
        "parol6.server.status_cache.get_fkine_flat_mm",
        lambda state: np.zeros((16,), dtype=float),
        raising=True,
    )

    # Patch enablement calculators to fixed values
    def _fake_joint(cache_self, q_rad):  # type: ignore[no-redef]
        bits = [1, 0] * 6
        cache_self.joint_en[:] = np.asarray(bits, dtype=np.uint8)
        cache_self._joint_en_ascii = ",".join(str(int(v)) for v in bits)

    def _fake_cart(cache_self, T, frame, q_rad):  # type: ignore[no-redef]
        bits = [1] * 12 if frame == "WRF" else [0] * 12
        ascii_bits = ",".join(str(b) for b in bits)
        if frame == "WRF":
            cache_self.cart_en_wrf[:] = np.asarray(bits, dtype=np.uint8)
            cache_self._cart_en_wrf_ascii = ascii_bits
        else:
            cache_self.cart_en_trf[:] = np.asarray(bits, dtype=np.uint8)
            cache_self._cart_en_trf_ascii = ascii_bits

    monkeypatch.setattr(StatusCache, "_compute_joint_enable", _fake_joint, raising=True)
    monkeypatch.setattr(StatusCache, "_compute_cart_enable", _fake_cart, raising=True)

    # Trigger an update with a fresh state
    state = ControllerState()
    # Change Position_in so StatusCache treats it as an update (pos_changed=True)
    arr = np.zeros((6,), dtype=np.int32)
    arr[0] = 1
    state.Position_in[:] = arr
    cache.update_from_state(state)

    txt = cache.to_ascii()
    assert "JOINT_EN=" in txt
    assert "CART_EN_WRF=" in txt and "CART_EN_TRF=" in txt
    assert "JOINT_EN=1,0,1,0,1,0,1,0,1,0,1,0" in txt
    assert "CART_EN_WRF=" + ",".join(["1"] * 12) in txt
    assert "CART_EN_TRF=" + ",".join(["0"] * 12) in txt
