from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]


def test_rsl_rl_play_can_record_deploy_cameras_until_first_reset():
    play = (REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py").read_text()

    assert '--record_deploy_cameras_until_reset' in play
    assert '--deploy_camera_output_dir' in play
    assert "if args_cli.video or args_cli.record_deploy_cameras_until_reset:" in play
    assert "args_cli.enable_cameras = True" in play
    assert "external_camera" in play
    assert "wrist_camera" in play
    assert "_record_deploy_camera_frame(" in play
    assert "_flush_deploy_camera_recordings(" in play
    assert "torch.any(dones)" in play
    assert "camera_recordings" in play
