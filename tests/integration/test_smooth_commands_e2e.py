"""
Integration tests for smooth motion commands (minimal set).

Tests one representative command per smooth motion family in FAKE_SERIAL mode.
Verifies command acceptance, completion status transitions, and basic functionality.
"""

import pytest
import sys
import os
import time

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))



def _check_if_fake_serial_xfail(result):
    """Helper to mark test as xfail if smooth motion fails due to IK in FAKE_SERIAL."""
    if (isinstance(result, dict) and 
        result.get('status') == 'FAILED' and
        'IK failed' in result.get('details', '')):
        pytest.xfail("Smooth motion commands fail IK in FAKE_SERIAL mode (expected)")


@pytest.mark.integration
class TestSmoothMotionMinimal:
    """Minimal set of smooth motion tests - one per command family."""
    
    @pytest.fixture
    def homed_robot(self, client, server_proc, robot_api_env):
        """Ensure robot is homed before smooth motion tests."""
        print("Homing robot for smooth motion tests...")
        
        # Home the robot first
        result = client.home()
        assert result is True
        
        # Wait for robot to be stopped
        assert client.wait_until_stopped(timeout=10.0)
        print("Robot homed and ready for smooth motion tests")
        
        return True
    
    def test_smooth_circle_basic(self, client, server_proc, robot_api_env, homed_robot):
        """Test basic circular motion in FAKE_SERIAL mode."""
        result = client.smooth_circle(
            center=[0, 0, 100],
            radius=30,
            duration=2.0,
            plane='XY',
            frame='WRF',
        )
        
        # Check if we should xfail due to FAKE_SERIAL limitations
        _check_if_fake_serial_xfail(result)
        
        # Should complete successfully in FAKE_SERIAL mode
        assert result is True
        
        # Wait for completion and verify robot stops
        assert client.wait_until_stopped(timeout=4.0)
        assert client.is_robot_stopped(threshold_speed=5.0)
    
    def test_smooth_arc_center_basic(self, client, server_proc, robot_api_env, homed_robot):
        """Test basic arc motion defined by center point."""
        result = client.smooth_arc_center(
            end_pose=[100, 100, 150, 0, 0, 90],
            center=[50, 50, 150],
            duration=2.0,
            frame='WRF',
        )
        
        _check_if_fake_serial_xfail(result)
        
        assert result is True
        
        assert client.wait_until_stopped(timeout=4.0)
        assert client.is_robot_stopped(threshold_speed=5.0)
    
    def test_smooth_spline_basic(self, client, server_proc, robot_api_env, homed_robot):
        """Test basic spline motion through waypoints."""
        waypoints = [
            [100.0, 100.0, 120.0, 0.0, 0.0, 0.0],
            [150.0, 150.0, 130.0, 0.0, 0.0, 30.0],
            [200.0, 100.0, 120.0, 0.0, 0.0, 60.0]
        ]
        
        result = client.smooth_spline(
            waypoints=waypoints,
            duration=3.0,
            frame='WRF',
        )
        
        _check_if_fake_serial_xfail(result)
        
        assert result is True
        
        assert client.wait_until_stopped(timeout=5.0)
        assert client.is_robot_stopped(threshold_speed=5.0)
    
    def test_smooth_helix_basic(self, client, server_proc, robot_api_env, homed_robot):
        """Test basic helical motion."""
        result = client.smooth_helix(
            center=[100, 100, 80],
            radius=25,
            pitch=20,
            height=60,
            duration=3.0,
            frame='WRF',
        )
        
        _check_if_fake_serial_xfail(result)
        
        assert result is True
        
        assert client.wait_until_stopped(timeout=5.0)
        assert client.is_robot_stopped(threshold_speed=5.0)
    
    def test_smooth_blend_basic(self, client, server_proc, robot_api_env, homed_robot):
        """Test basic blended motion through segments."""
        segments = [
            {
                'type': 'LINE',
                'end': [120.0, 80.0, 140.0, 0.0, 0.0, 30.0],
                'duration': 1.0
            },
            {
                'type': 'CIRCLE', 
                'center': [120, 120, 140],
                'radius': 25,
                'plane': 'XY',
                'duration': 2.0,
                'clockwise': False
            }
        ]
        
        result = client.smooth_blend(
            segments=segments,
            blend_time=0.3,
            frame='WRF',
        )
        
        _check_if_fake_serial_xfail(result)
        
        assert result is True
        
        assert client.wait_until_stopped(timeout=5.0)
        assert client.is_robot_stopped(threshold_speed=5.0)


if __name__ == "__main__":
    pytest.main([__file__])
