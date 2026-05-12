from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "scripts_peg_insertion" / "07_train_all_easy_from_scratch.sh"


def test_train_all_easy_from_scratch_uses_all_reset_types_without_resume():
    script = SCRIPT.read_text()

    assert "--resume_path" not in script
    assert "DEFAULT_RESUME_PATH" not in script
    assert "OmniReset-Arx5-OSC-State-v0" in script
    assert "ObjectAnywhereEEAnywhere" in script
    assert "ObjectRestingEEGrasped" in script
    assert "ObjectAnywhereEEGrasped" in script
    assert "ObjectPartiallyAssembledEEGrasped" in script
    assert 'PROBS="[0.25,0.25,0.25,0.25]"' in script
    assert 'MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"' in script
    assert 'SUCCESS_THRESHOLD_SCALE="${SUCCESS_THRESHOLD_SCALE:-4.0}"' in script
    assert "env.commands.task_command.success_threshold_scale" in script


def test_train_all_easy_from_scratch_checks_required_local_datasets():
    script = SCRIPT.read_text()

    assert "Datasets/OmniReset" in script
    assert "Grasps/Peg/grasps.pt" in script
    assert "Resets/Peg__PegHole/partial_assemblies.pt" in script
    assert "resets_ObjectAnywhereEEAnywhere.pt" in script
    assert "resets_ObjectRestingEEGrasped.pt" in script
    assert "resets_ObjectAnywhereEEGrasped.pt" in script
    assert "resets_ObjectPartiallyAssembledEEGrasped.pt" in script
