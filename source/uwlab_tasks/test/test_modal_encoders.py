from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).parents[3]
MODAL_ENCODERS_PATH = REPO_ROOT / "source" / "uwlab_rl" / "uwlab_rl" / "rsl_rl" / "modal_encoders.py"
VISION_ACTOR_CRITIC_PATH = REPO_ROOT / "source" / "uwlab_rl" / "uwlab_rl" / "rsl_rl" / "vision_actor_critic.py"
MODAL_ACTOR_CRITIC_PATH = REPO_ROOT / "source" / "uwlab_rl" / "uwlab_rl" / "rsl_rl" / "modal_vision_actor_critic.py"


def _load_modal_encoders():
    spec = importlib.util.spec_from_file_location("uwlab_modal_encoders_for_test", MODAL_ENCODERS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_modal_actor_critic():
    package = types.ModuleType("uwlab_rl.rsl_rl")
    package.__path__ = [str(MODAL_ENCODERS_PATH.parent)]
    sys.modules.setdefault("uwlab_rl", types.ModuleType("uwlab_rl"))
    sys.modules["uwlab_rl.rsl_rl"] = package
    _load_module("uwlab_rl.rsl_rl.modal_encoders", MODAL_ENCODERS_PATH)
    _load_module("uwlab_rl.rsl_rl.vision_actor_critic", VISION_ACTOR_CRITIC_PATH)
    module = _load_module("uwlab_rl.rsl_rl.modal_vision_actor_critic", MODAL_ACTOR_CRITIC_PATH)
    return module.ModalVisionActorCritic


def test_bbox_to_confidence_mask_uses_confidence_and_ignores_padding():
    modal = _load_modal_encoders()
    bboxes = torch.tensor(
        [[[[1.0, 1.0, 3.0, 3.0, 0.7], [-1.0, -1.0, -1.0, -1.0, -1.0]]]],
        dtype=torch.float32,
    )

    mask = modal.bbox_to_confidence_mask(bboxes, image_size=4)

    assert mask.shape == (1, 1, 1, 4, 4)
    assert torch.allclose(mask[0, 0, 0, 1:3, 1:3], torch.full((2, 2), 0.7))
    assert mask[0, 0, 0, 0, 0].item() == 0.0


def test_depth_and_mask_patch_encoders_are_configurable():
    modal = _load_modal_encoders()
    depth_encoder = modal.DepthMapEncoder(image_size=128, patch_size=8, embed_dim=64)
    mask_encoder = modal.MaskPatchEncoderWithoutRGB(image_size=128, patch_size=8, embed_dim=64)

    depth_out = depth_encoder(torch.rand(3, 1, 128, 128))
    mask_out = mask_encoder(torch.rand(3, 1, 128, 128))

    assert depth_encoder.num_patches == 64
    assert mask_encoder.num_patches == 64
    assert depth_out.shape == (3, 64, 8, 8)
    assert mask_out.shape == (3, 64, 8, 8)


def test_ssi_style_modal_encoder_outputs_cls_feature():
    modal = _load_modal_encoders()
    encoder = modal.SSIStyleModalEncoder(
        num_views=2,
        image_size=128,
        max_bbox_num=3,
        track_len=16,
        num_track_ids=32,
        embed_dim=64,
        cross_attention={"num_heads": 4, "dropout": 0.0},
        spatial_transformer={"num_layers": 2, "num_heads": 4, "ffn_dim": 128, "dropout": 0.0},
    )
    encoder.eval()
    depth_map = torch.rand(2, 2, 1, 128, 128)
    bboxes = torch.tensor(
        [
            [
                [[10.0, 10.0, 40.0, 40.0, 0.9], [60.0, 60.0, 90.0, 90.0, 0.8], [-1.0, -1.0, -1.0, -1.0, -1.0]],
                [[20.0, 20.0, 50.0, 50.0, 0.9], [70.0, 70.0, 100.0, 100.0, 0.8], [-1.0, -1.0, -1.0, -1.0, -1.0]],
            ],
            [
                [[12.0, 8.0, 38.0, 42.0, 0.7], [58.0, 62.0, 88.0, 92.0, 0.6], [-1.0, -1.0, -1.0, -1.0, -1.0]],
                [[18.0, 22.0, 48.0, 52.0, 0.7], [68.0, 72.0, 98.0, 102.0, 0.6], [-1.0, -1.0, -1.0, -1.0, -1.0]],
            ],
        ],
        dtype=torch.float32,
    )
    trajectory = torch.rand(2, 2, 16, 32, 2)

    out = encoder(depth_map, bboxes, trajectory)

    assert out.shape == (2, 64)
    assert encoder.shapes.num_visual_tokens == 128
    assert encoder.shapes.num_track_tokens == 64
    assert encoder.shapes.num_total_tokens == 193


def test_ssi_style_modal_encoder_default_output_dim_is_256():
    modal = _load_modal_encoders()
    encoder = modal.SSIStyleModalEncoder(spatial_transformer={"num_layers": 1})

    assert encoder.output_dim == 256
    assert encoder.shapes.embed_dim == 256


def test_modal_vision_actor_critic_rsl_rl_api_with_cached_modal_obs():
    tensordict = pytest.importorskip("tensordict")
    ModalVisionActorCritic = _load_modal_actor_critic()

    batch_size = 2
    obs = tensordict.TensorDict(
        {
            "policy": tensordict.TensorDict(
                {
                    "depth_map": torch.rand(batch_size, 2, 1, 128, 128),
                    "bboxes": torch.tensor(
                        [
                            [
                                [[10.0, 10.0, 40.0, 40.0, 0.9], [60.0, 60.0, 90.0, 90.0, 0.8], [-1.0, -1.0, -1.0, -1.0, -1.0]],
                                [[20.0, 20.0, 50.0, 50.0, 0.9], [70.0, 70.0, 100.0, 100.0, 0.8], [-1.0, -1.0, -1.0, -1.0, -1.0]],
                            ],
                            [
                                [[12.0, 8.0, 38.0, 42.0, 0.7], [58.0, 62.0, 88.0, 92.0, 0.6], [-1.0, -1.0, -1.0, -1.0, -1.0]],
                                [[18.0, 22.0, 48.0, 52.0, 0.7], [68.0, 72.0, 98.0, 102.0, 0.6], [-1.0, -1.0, -1.0, -1.0, -1.0]],
                            ],
                        ],
                        dtype=torch.float32,
                    ),
                    "trajectory": torch.rand(batch_size, 2, 16, 32, 2),
                },
                batch_size=[batch_size],
            ),
            "critic": torch.rand(batch_size, 11),
        },
        batch_size=[batch_size],
    )

    actor_critic = ModalVisionActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions=7,
        modal_encoder={
            "embed_dim": 64,
            "cross_attention": {"num_heads": 4, "dropout": 0.0},
            "spatial_transformer": {"num_layers": 1, "num_heads": 4, "ffn_dim": 128, "dropout": 0.0},
        },
        use_joint_pos=False,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        noise_std_type="log",
    )
    actor_critic.eval()

    action = actor_critic.act(obs)
    inference_action = actor_critic.act_inference(obs)
    value = actor_critic.evaluate(obs)
    log_prob = actor_critic.get_actions_log_prob(action)

    assert action.shape == (batch_size, 7)
    assert inference_action.shape == (batch_size, 7)
    assert value.shape == (batch_size, 1)
    assert log_prob.shape == (batch_size,)
