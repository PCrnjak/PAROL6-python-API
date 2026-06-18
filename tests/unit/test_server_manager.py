import pytest

from parol6 import Robot


def test_is_available_false_when_no_server(free_port):
    robot = Robot(port=free_port)
    assert not robot.is_available()


def test_robot_start_and_stop(free_port):
    host = "127.0.0.1"

    robot = Robot(host=host, port=free_port, timeout=15.0)
    assert not robot.is_available()

    try:
        robot.start()
        assert robot.is_available()
    finally:
        robot.stop()

    assert not robot.is_available()


def test_robot_start_fast_fails_when_already_running(free_port):
    host = "127.0.0.1"

    robot = Robot(host=host, port=free_port, timeout=15.0)
    try:
        robot.start()
        assert robot.is_available()

        robot2 = Robot(host=host, port=free_port)
        with pytest.raises(RuntimeError):
            robot2.start()
    finally:
        robot.stop()
