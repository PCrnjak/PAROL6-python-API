"""
Integration tests for enablement detection via IK worker subprocess.
"""

import time

import numpy as np
import pytest

from parol6.server.status_cache import StatusCache
import parol6.PAROL6_ROBOT as PAROL6_ROBOT


@pytest.mark.integration
def test_ik_worker_detects_joint_limits():
    """
    Test that the IK worker correctly detects when a joint is near its limits.

    Joint enablement format: [J1+, J1-, J2+, J2-, ..., J6+, J6-]
    - When near max limit: + direction disabled (0), - direction enabled (1)
    - When near min limit: + direction enabled (1), - direction disabled (0)
    """
    cache = StatusCache()

    try:
        time.sleep(0.5)  # Allow more time for subprocess startup on Windows
        assert cache._ik_process.is_alive(), "IK worker failed to start"

        # Start from home position and move J1 near its max limit
        from parol6.config import HOME_ANGLES_DEG

        qlim = PAROL6_ROBOT.robot.qlim

        q = np.deg2rad(HOME_ANGLES_DEG)
        # Delta is 0.2 degrees = 0.0035 rad, so we need to be within that
        q[0] = qlim[1, 0] - 0.001  # J1 very near max limit

        T = PAROL6_ROBOT.robot.fkine(q)
        T_matrix = T

        cache._submit_ik_request(q, T_matrix)

        ready = False
        for _ in range(200):  # Longer timeout for CI - IK worker does 24 IK solves
            ready = cache._poll_ik_results()
            if ready:
                break
            time.sleep(0.02)

        assert ready, "IK worker did not return results"
        joint_en = cache.joint_en

        # J1 near max: J1+ should be disabled, J1- should be enabled
        assert joint_en[0] == 0, (
            f"J1+ should be disabled near max limit, got {joint_en[0]}"
        )
        assert joint_en[1] == 1, (
            f"J1- should be enabled near max limit, got {joint_en[1]}"
        )

        # Now test J1 near min limit
        q[0] = qlim[0, 0] + 0.001  # J1 very near min limit

        T = PAROL6_ROBOT.robot.fkine(q)
        T_matrix = T

        cache._submit_ik_request(q, T_matrix)

        ready = False
        for _ in range(200):  # Longer timeout for CI - IK worker does 24 IK solves
            ready = cache._poll_ik_results()
            if ready:
                break
            time.sleep(0.02)

        assert ready, "IK worker did not return results for min limit test"
        joint_en = cache.joint_en

        # J1 near min: J1+ should be enabled, J1- should be disabled
        assert joint_en[0] == 1, (
            f"J1+ should be enabled near min limit, got {joint_en[0]}"
        )
        assert joint_en[1] == 0, (
            f"J1- should be disabled near min limit, got {joint_en[1]}"
        )

    finally:
        cache.close()


@pytest.mark.integration
def test_ik_worker_all_enabled_in_safe_position():
    """
    Test that all directions are enabled when robot is in the true center of its limits.
    """
    cache = StatusCache()

    try:
        time.sleep(0.5)  # Allow more time for subprocess startup on Windows
        assert cache._ik_process.is_alive()

        # Use home position - a known safe position
        from parol6.config import HOME_ANGLES_DEG

        q_home = np.deg2rad(HOME_ANGLES_DEG)

        T = PAROL6_ROBOT.robot.fkine(q_home)
        T_matrix = T

        cache._submit_ik_request(q_home, T_matrix)

        ready = False
        for _ in range(200):  # Longer timeout for CI - IK worker does 24 IK solves
            ready = cache._poll_ik_results()
            if ready:
                break
            time.sleep(0.02)

        assert ready, "IK worker did not return results in time"
        joint_en = cache.joint_en

        # All joint directions should be enabled in true center position
        assert np.all(joint_en == 1), (
            f"All joints should be enabled at center, got {joint_en}"
        )

    finally:
        cache.close()
