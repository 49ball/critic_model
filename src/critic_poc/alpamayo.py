"""Optional NVIDIA Alpamayo 1.5 boundary. NVIDIA packages are imported lazily."""

import numpy as np
from .schema import validate_scene, write_scenes


def normalize_predictions(xyz, rotations):
    """Official OSS layout: [B=1, trajectory_group=1, K, T, 3]."""
    xyz, rotations = np.asarray(xyz), np.asarray(rotations)
    if (
        xyz.ndim != 5
        or xyz.shape[:2] != (1, 1)
        or xyz.shape[-1] != 3
        or rotations.shape != xyz.shape[:-1] + (3, 3)
    ):
        raise ValueError("Expected xyz [1,1,K,T,3], rotations [1,1,K,T,3,3]")
    if not np.isfinite(xyz).all() or not np.isfinite(rotations).all():
        raise ValueError("Alpamayo output contains nonfinite values")
    rot = rotations[0, 0]
    if not np.allclose(rot @ np.swapaxes(rot, -1, -2), np.eye(3), atol=1e-3) or not np.allclose(
        np.linalg.det(rot), 1, atol=1e-3
    ):
        raise ValueError("Invalid rotation matrix")
    yaw = np.arctan2(rot[..., 1, 0], rot[..., 0, 0])
    return np.concatenate([xyz[0, 0, ..., :2], yaw[..., None]], axis=-1)


def merge_world(scene, world):
    """Sidecar must explicitly match the clip, time, and ego coordinate frame."""
    for field in ("scene_id", "t0_us", "frame"):
        if world.get(field) != scene[field]:
            raise ValueError(f"World sidecar {field} mismatch")
    result = dict(scene)
    for key in ("agents", "observation_complete", "lane_half_width"):
        if key in world:
            result[key] = world[key]
    # Vehicle dimensions/desire can be overridden; speed is derived from observed history.
    result["ego"] = dict(scene["ego"])
    for key in ("length", "width", "desired_speed"):
        if key in world.get("ego", {}):
            result["ego"][key] = world["ego"][key]
    return validate_scene(result)


def export_alpamayo(
    clip_id,
    t0_us,
    out,
    samples=8,
    seed=42,
    world=None,
    model_id="nvidia/Alpamayo-1.5-10B",
    revision=None,
    ego_length=4.8,
    ego_width=2.0,
):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Alpamayo export requires NVIDIA CUDA GPU; use synthetic for CPU demo.")
    if not 1 <= samples <= 128:
        raise ValueError("samples must be between 1 and 128")
    try:
        from alpamayo1_5 import helper
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    except ImportError as exc:
        raise RuntimeError(
            "Install official NVlabs/alpamayo1.5 in its Python 3.12 environment; "
            "see docs/alpamayo.md."
        ) from exc
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    options = {"dtype": torch.bfloat16}
    if revision:
        options["revision"] = revision
    model = Alpamayo1_5.from_pretrained(model_id, **options).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = helper.to_device(
        {
            "tokenized_data": tokenized,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )
    torch.manual_seed(seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        xyz, rot, _ = model.sample_trajectories_from_data_with_vlm_rollout(
            data=inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=samples,
            max_generation_length=256,
            return_extra=True,
        )
    candidates = normalize_predictions(xyz.float().cpu().numpy(), rot.float().cpu().numpy())
    history = data["ego_history_xyz"][0, 0].float().cpu().numpy()
    speed = float(np.linalg.norm(history[-1, :2] - history[-2, :2]) / 0.1)
    # The official loader also returns future ego labels; deliberately never exported as input.
    scene = {
        "schema_version": 1,
        "scene_id": f"{clip_id}:{t0_us}",
        "group_id": clip_id,
        "domain": "physical_ai_av",
        "frame": "ego_t0",
        "t0_us": t0_us,
        "dt": 0.1,
        "ego": {
            "speed": speed,
            "length": ego_length,
            "width": ego_width,
            "desired_speed": max(speed, 2.0),
        },
        "agents": [],
        "observation_complete": False,
        "lane_half_width": None,
        "candidates": candidates.tolist(),
        "provenance": {
            "generator": "alpamayo1.5",
            "model_id": model_id,
            "revision": revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "seed": seed,
            "world_source": "none",
            "vehicle_dimensions": "user_supplied_or_default",
        },
    }
    if world is not None:
        scene = merge_world(scene, world)
        scene["provenance"]["world_source"] = "user_sidecar"
    write_scenes(out, [scene])
    return scene
