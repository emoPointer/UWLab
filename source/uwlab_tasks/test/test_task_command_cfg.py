from pathlib import Path


def test_task_command_cfg_exposes_success_threshold_scale():
    commands_cfg = (
        Path(__file__).parents[1]
        / "uwlab_tasks"
        / "manager_based"
        / "manipulation"
        / "omnireset"
        / "mdp"
        / "commands_cfg.py"
    ).read_text()

    assert "success_threshold_scale: float = 1.0" in commands_cfg


def test_task_command_uses_success_threshold_scale():
    commands = (
        Path(__file__).parents[1]
        / "uwlab_tasks"
        / "manager_based"
        / "manipulation"
        / "omnireset"
        / "mdp"
        / "commands.py"
    ).read_text()

    assert "cfg.success_threshold_scale" in commands
    assert "success_position_threshold" in commands
    assert "success_orientation_threshold" in commands
