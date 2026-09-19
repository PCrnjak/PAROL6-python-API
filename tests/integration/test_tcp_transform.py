"""TCP calibration configuration through real client and controller paths."""

import socket

import msgspec
import numpy as np
import pytest
from waldoctl.setup import Pose

from parol6.client.async_client import AsyncRobotClient
from parol6.protocol.wire import CmdType, MsgType


@pytest.mark.asyncio
async def test_unanswered_tcp_readback_cannot_clear_a_saved_calibration():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as silent:
        silent.bind(("127.0.0.1", 0))
        client = AsyncRobotClient(port=silent.getsockname()[1], timeout=0.05, retries=0)
        try:
            with pytest.raises(TimeoutError):
                await client.tcp_offset()
            with pytest.raises(TimeoutError):
                await client.tcp_transform()
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_full_tcp_transform_agrees_across_wire_fk_preview_and_motion(ports):
    from parol6.client.dry_run_client import DryRunRobotClient
    from parol6.robot import Robot

    values = (5.0, -3.0, 20.0, 20.0, 25.0, -10.0)
    async with AsyncRobotClient(host=ports.server_ip, port=ports.server_port) as client:
        base = await client.pose()
        angles = await client.angles()
        assert base is not None and angles is not None
        expected = Pose(tuple(base)).matrix() @ Pose(values).matrix()
        index = await client.set_tcp_transform(*values)
        assert index >= 0 and await client.wait_command(index, timeout=10)
        assert await client.tcp_transform() == pytest.approx(values)
        assert await client.tcp_offset() == pytest.approx(values[:3])
        assert await client.wait_status(
            lambda s: np.allclose(
                np.asarray(s.pose).reshape(4, 4), expected, atol=0.05
            ),
            timeout=5,
        ), "stationary STATUS did not adopt the full TCP transform"

        local = Robot()
        local.set_active_tool(
            "NONE",
            tcp_offset_m=tuple(v / 1000 for v in values[:3]),
            tcp_rotation_rad=tuple(np.radians(values[3:])),
        )
        local_pose = local.fk(np.radians(angles), np.empty(6))
        local_matrix = Pose(
            tuple([*(local_pose[:3] * 1000), *np.degrees(local_pose[3:])])
        ).matrix()
        assert local_matrix == pytest.approx(expected, abs=0.05)

        preview = DryRunRobotClient(initial_joints_deg=angles)
        preview.set_tcp_transform(*values)
        predicted = preview.move_l([0, 0, 5, 0, 0, 0], frame="TRF", rel=True, speed=0.2)
        assert predicted is not None and predicted.error is None
        target = expected @ Pose((0, 0, 5, 0, 0, 0)).matrix()
        predicted_pose = predicted.tcp_poses[-1]
        predicted_matrix = Pose(
            tuple([*(predicted_pose[:3] * 1000), *np.degrees(predicted_pose[3:])])
        ).matrix()
        assert predicted_matrix[:3, 3] == pytest.approx(target[:3, 3], abs=1.0)

        index = await client.move_l(
            [0, 0, 5, 0, 0, 0], frame="TRF", rel=True, speed=0.2
        )
        assert index >= 0 and await client.wait_command(index, timeout=15)
        actual = await client.pose()
        assert actual is not None
        actual_matrix = Pose(tuple(actual)).matrix()
        assert actual_matrix[:3, 3] == pytest.approx(target[:3, 3], abs=1.0)
        assert actual_matrix[:3, :3] == pytest.approx(target[:3, :3], abs=0.02)

        delay = await client.delay(5)
        assert await client.wait_status(lambda s: s.executing_index == delay, timeout=5)
        pending = await client.set_tcp_transform(0, 0, 40, 0, 90, 0)
        assert pending > delay
        assert await client.tcp_transform() == pytest.approx(values)
        assert await client.stop() > 0
        assert await client.tcp_transform() == pytest.approx(values)
        index = await client.move_l(actual, speed=0.2)
        assert index >= 0 and await client.wait_command(index, timeout=15), (
            "cancelled calibration leaked into the planner"
        )

        index = await client.select_tool("NONE")
        assert await client.wait_command(index, timeout=10)
        assert await client.tcp_transform() == pytest.approx(values)
        index = await client.set_tcp_offset(1, 2, 3)
        assert await client.wait_command(index, timeout=10)
        assert await client.tcp_transform() == pytest.approx([1, 2, 3, 0, 0, 0])

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as raw:
            raw.settimeout(2)
            for slot in range(6):
                for invalid in (float("nan"), float("inf"), float("-inf")):
                    payload = [float(v) for v in values]
                    payload[slot] = invalid
                    raw.sendto(
                        msgspec.msgpack.encode(
                            [int(CmdType.SET_TCP_TRANSFORM), *payload]
                        ),
                        (ports.server_ip, ports.server_port),
                    )
                    reply = msgspec.msgpack.decode(raw.recv(65535))
                    assert reply[0] == int(MsgType.ERROR)
        assert await client.tcp_transform() == pytest.approx([1, 2, 3, 0, 0, 0])
        index = await client.select_tool("SSG-48")
        assert await client.wait_command(index, timeout=10)
        assert await client.tcp_transform() == pytest.approx([0] * 6)


@pytest.mark.asyncio
async def test_tcp_calibration_binding_reports_the_selected_tool_variant(ports):
    async with AsyncRobotClient(host=ports.server_ip, port=ports.server_port) as client:
        for variant in ("vertical", "horizontal"):
            index = await client.select_tool("PNEUMATIC", variant_key=variant)
            assert await client.wait_command(index, timeout=10)
            observed = []

            def capture(status):
                observed.append(status.tool_status.variant_key)
                return len(observed) > 1 and status.tool_status.key == "PNEUMATIC"

            assert await client.wait_status(capture, timeout=3)
            assert observed[-1] == variant
            tool = await client.tool.status()
            assert tool is not None and tool.variant_key == variant
