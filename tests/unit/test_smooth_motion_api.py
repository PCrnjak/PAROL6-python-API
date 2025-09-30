import inspect
import importlib


def test_smooth_motion_reexports_exist():
    sm = importlib.import_module("parol6.smooth_motion")

    # Required re-exports
    for name in [
        "CircularMotion",
        "SplineMotion",
        "HelixMotion",
        "WaypointTrajectoryPlanner",
    ]:
        assert hasattr(sm, name), f"parol6.smooth_motion missing {name}"
        obj = getattr(sm, name)
        assert inspect.isclass(obj), f"{name} should be a class"

    # Optional blenders (presence depends on legacy module)
    # If present, they should be classes as well
    for opt_name in ["MotionBlender", "AdvancedMotionBlender"]:
        if hasattr(sm, opt_name):
            opt = getattr(sm, opt_name)
            assert inspect.isclass(opt), f"{opt_name} should be a class if present"
