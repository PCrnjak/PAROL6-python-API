"""Per-joint drive faults ride the status broadcast.

The drivers report fault bits over the serial link rather than analog
temperature or current registers, so this backend's drive health is faults
and nothing else — and an all-clear has to be distinguishable from a
backend that reports no faults at all, or a display cannot tell "healthy"
from "not instrumented".

Driven through the real cache and the real codec: a ControllerState carrying
the bits the firmware would set, encoded by the cache the broadcaster uses,
decoded by the buffer the client fills.
"""

import pytest

from parol6.protocol.wire import StatusBuffer, decode_status_bin_into
from parol6.server.state import ControllerState
from parol6.server.status_cache import StatusCache


def _decode(cache: StatusCache) -> StatusBuffer:
    buf = StatusBuffer()
    assert decode_status_bin_into(cache.to_binary(), buf), "STATUS failed to decode"
    return buf


@pytest.mark.integration
def test_faults_reach_the_client_against_the_joint_that_tripped():
    """A tripped drive names its condition, on its own joint, and a healthy
    bus still reports one entry per joint so all-clear is visible as such."""
    cache = StatusCache()
    try:
        state = ControllerState()

        cache.update_from_state(state)
        healthy = _decode(cache).drive_health["faults"]
        assert healthy == [(), (), (), (), (), ()], (
            "a healthy bus must still report one entry per joint, so that "
            f"all-clear is distinguishable from no reporting: {healthy}"
        )

        state.Temperature_error_in[2] = 1
        state.Position_error_in[4] = 1
        cache.update_from_state(state)
        faults = _decode(cache).drive_health["faults"]
        assert faults[2] == ("overtemperature",), faults
        assert faults[4] == ("following_error",), faults
        assert all(faults[i] == () for i in (0, 1, 3, 5)), (
            f"a fault on one drive must not read as a fault on another: {faults}"
        )

        # Both conditions on one drive, and the earlier fault clearing.
        state.Temperature_error_in[2] = 0
        state.Temperature_error_in[4] = 1
        cache.update_from_state(state)
        faults = _decode(cache).drive_health["faults"]
        assert faults[2] == (), "a cleared fault must stop being reported"
        assert set(faults[4]) == {"overtemperature", "following_error"}, faults
    finally:
        cache.close()


@pytest.mark.integration
def test_a_bus_with_no_analog_registers_reports_no_readings():
    """Faults are this backend's only drive health. Empty temperature and
    current lists are what tell a consumer there is no such sensor, rather
    than a row of zeros that reads as a cold, idle drive."""
    cache = StatusCache()
    try:
        cache.update_from_state(ControllerState())
        health = _decode(cache).drive_health
        assert health.get("faults"), "faults are reported"
        assert not health.get("temperatures_c"), health
        assert not health.get("currents_ma"), health
        assert health.get("bus_voltage_v") is None, health
    finally:
        cache.close()
