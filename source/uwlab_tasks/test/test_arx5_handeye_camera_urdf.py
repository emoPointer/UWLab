from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]


def test_arx5_colored_urdf_contains_handeye_camera_link():
    urdf = (
        REPO_ROOT
        / "source"
        / "uwlab_assets"
        / "uwlab_assets"
        / "robots"
        / "arx5"
        / "assets"
        / "arx5_colored.urdf"
    ).read_text()

    assert '<link name="handeye_camera"/>' in urdf
    assert '<joint name="handeye_camera_joint" type="fixed">' in urdf
    assert '<parent link="link6" />' in urdf
    assert '<child link="handeye_camera" />' in urdf
    assert '<origin xyz="0.064363 0.004000 0.077500" rpy="-0.7528672 0.00210611 1.57210039" />' in urdf
    assert "fovy = 42.47 deg" in urdf
