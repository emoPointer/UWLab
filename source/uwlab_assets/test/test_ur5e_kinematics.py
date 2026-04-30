import importlib.util
import sys
import types
from pathlib import Path


def test_ur5e_calibration_metadata_resolves_from_uwlab_cache(tmp_path, monkeypatch):
    metadata_dir = (
        tmp_path
        / ".cache"
        / "uwlab"
        / "assets"
        / "Robots"
        / "UniversalRobots"
        / "Ur5e2f85RobotiqGripperCalibrated"
    )
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "calibrated_joints:",
                "  xyz:",
                *["    - [0.0, 0.0, 0.0]" for _ in range(6)],
                "  rpy:",
                *["    - [0.0, 0.0, 0.0]" for _ in range(6)],
                "link_inertials:",
                "  masses: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]",
                "  coms:",
                *["    - [0.0, 0.0, 0.0]" for _ in range(6)],
                "  inertias:",
                *["    - [1.0, 1.0, 1.0]" for _ in range(6)],
            ]
        )
    )

    package_name = "uwlab_assets.robots.ur5e_robotiq_gripper"
    fake_package = types.ModuleType(package_name)
    fake_package.__path__ = []
    fake_robot_module = types.ModuleType(f"{package_name}.ur5e_robotiq_2f85_gripper")
    fake_robot_module.UR5E_ARTICULATION = types.SimpleNamespace(
        spawn=types.SimpleNamespace(
            usd_path=(
                "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/"
                "Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/"
                "ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd"
            )
        )
    )
    monkeypatch.setitem(sys.modules, package_name, fake_package)
    monkeypatch.setitem(sys.modules, fake_robot_module.__name__, fake_robot_module)

    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab_utils = types.ModuleType("isaaclab.utils")
    fake_isaaclab_assets = types.ModuleType("isaaclab.utils.assets")
    fake_isaaclab_assets.retrieve_file_path = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("UR5e calibration should resolve through UWLab local asset cache")
    )
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", fake_isaaclab_utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.assets", fake_isaaclab_assets)

    module_path = (
        Path(__file__).parents[1]
        / "uwlab_assets"
        / "robots"
        / "ur5e_robotiq_gripper"
        / "kinematics.py"
    )
    spec = importlib.util.spec_from_file_location(f"{package_name}.kinematics", module_path)
    assert spec is not None
    assert spec.loader is not None
    kinematics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kinematics)

    def fail_if_isaaclab_resolver_is_used(*_args, **_kwargs):
        raise AssertionError("UR5e calibration should resolve through UWLab local asset cache")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(kinematics, "retrieve_file_path", fail_if_isaaclab_resolver_is_used, raising=False)
    kinematics._load_calibration.cache_clear()

    calibration = kinematics._load_calibration()

    assert calibration["joints_xyz"].shape == (6, 3)
    assert calibration["joints_rpy"].shape == (6, 3)
    assert calibration["link_masses"].shape == (6,)
