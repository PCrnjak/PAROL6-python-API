import pytest
from unittest.mock import AsyncMock

from parol6 import RobotClient


def _pose_payload_from_matrix(m):
    # Flatten list of lists to comma string after prefix
    flat = []
    for row in m:
        flat.extend(row)
    return "POSE|" + ",".join(str(x) for x in flat)


def test_get_pose_rpy_identity_translation(monkeypatch):
    """
    Validate get_pose_rpy converts 4x4 pose matrix to [x,y,z,rx,ry,rz] (mm,deg).
    Use identity rotation with translation (10,20,30) mm.
    """
    client = RobotClient()

    # Identity rotation with translation in last column (row-major)
    mat = [
        [1, 0, 0, 10],
        [0, 1, 0, 20],
        [0, 0, 1, 30],
        [0, 0, 0, 1],
    ]
    payload = _pose_payload_from_matrix(mat)

    # Patch the async client's _request coroutine used under the hood
    mock_request = AsyncMock(return_value=payload)
    monkeypatch.setattr(client.async_client, "_request", mock_request)

    pose_rpy = client.get_pose_rpy()
    assert pose_rpy is not None
    # Translations
    assert pose_rpy[0:3] == [10, 20, 30]
    # Identity rotation -> zero Euler angles (within tolerance)
    rx, ry, rz = pose_rpy[3:6]
    assert abs(rx) < 1e-6
    assert abs(ry) < 1e-6
    assert abs(rz) < 1e-6


def test_get_pose_rpy_malformed_payload(monkeypatch):
    """
    Malformed POSE payload (wrong length) should return None.
    """
    client = RobotClient()

    # Not 16 elements
    mock_request = AsyncMock(return_value="POSE|1,2,3")
    monkeypatch.setattr(client.async_client, "_request", mock_request)

    pose_rpy = client.get_pose_rpy()
    assert pose_rpy is None
