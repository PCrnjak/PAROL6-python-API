"""Attachment declarations gate on the six joints, not the padded homed byte."""

import numpy as np
from waldoctl import Sphere

from parol6.server.state import ControllerState


def test_attachments_accept_a_homed_arm_with_unused_homed_slots_clear():
    state = ControllerState()
    state.enabled = True
    # The firmware byte carries six joints; slots 6-7 are always zero on
    # hardware (the fake serial path fills all eight).
    state.Homed_in[:] = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.uint8)
    part = Sphere(name="part", radius=0.02).attach(
        flange_pose=(0.0, 0.0, 0.1, 0.0, 0.0, 0.0), epoch=state.attachment_epoch
    )
    state.set_shapes([part])
    assert state.has_attachments and state.attachments_valid
