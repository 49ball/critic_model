"""Command line entry points. Run python -m critic_poc --help."""

import argparse
import json
from pathlib import Path
import random

from .schema import read_scenes, write_scenes, write_json


def parser():
    root = argparse.ArgumentParser(
        description="Alpamayo Critic PoC: generate / label / train / evaluate / rank"
    )
    commands = root.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic", help="Generate toy JSONL splits")
    synthetic.add_argument("--out", required=True)
    synthetic.add_argument("--scenes", type=int, default=400)
    synthetic.add_argument("--candidates", type=int, default=8)
    synthetic.add_argument("--steps", type=int, default=32)
    synthetic.add_argument("--seed", type=int, default=7)

    label = commands.add_parser("label", help="Create explicit CV/corridor weak labels")
    label.add_argument("--data", required=True)
    label.add_argument("--out", required=True)
    label.add_argument(
        "--replace-labels",
        action="store_true",
        help="Explicitly replace existing labels with rule weak labels",
    )

    split = commands.add_parser("split", help="Split a JSONL file by group_id")
    split.add_argument("--data", required=True)
    split.add_argument("--out", required=True)
    split.add_argument("--seed", type=int, default=7)

    train = commands.add_parser("train", help="Train a multi-head critic")
    train.add_argument("--train", required=True)
    train.add_argument("--val", required=True)
    train.add_argument("--calibration")
    train.add_argument("--out", required=True)
    train.add_argument("--config", help="JSON config; see configs/default.json")
    train.add_argument("--epochs", type=int)
    train.add_argument("--device", default="cpu")

    evaluate = commands.add_parser("evaluate", help="Evaluate an unseen labeled split")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--allow-seen-groups", action="store_true")
    evaluate.add_argument("--allow-domain-shift", action="store_true")
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--device", default="cpu")

    rank = commands.add_parser("rank", help="Rank one JSONL scene, abstain when checks unavailable")
    rank.add_argument("--checkpoint", required=True)
    rank.add_argument("--data", required=True)
    rank.add_argument("--scene-index", type=int, default=0)
    rank.add_argument("--out", required=True)
    rank.add_argument("--mc-samples", type=int, default=8)
    rank.add_argument("--threshold", type=float, default=0.5)
    rank.add_argument("--max-uncertainty", type=float, default=0.15)
    rank.add_argument("--allow-domain-shift", action="store_true")
    rank.add_argument("--device", default="cpu")

    export = commands.add_parser("alpamayo-export", help="Run real Alpamayo 1.5 on a CUDA host")
    export.add_argument("--clip-id", required=True)
    export.add_argument("--t0-us", type=int, default=5100000)
    export.add_argument("--out", required=True)
    export.add_argument("--samples", type=int, default=8)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--world-json", help="Observed actors / straight corridor sidecar")
    export.add_argument("--model-id", default="nvidia/Alpamayo-1.5-10B")
    export.add_argument("--revision", help="Pin Hugging Face model revision")
    export.add_argument("--ego-length", type=float, default=4.8)
    export.add_argument("--ego-width", type=float, default=2.0)

    demo = commands.add_parser("demo", help="CPU end-to-end synthetic training and reports")
    demo.add_argument("--out", required=True)
    demo.add_argument("--epochs", type=int, default=5)
    demo.add_argument("--scenes", type=int, default=200)
    demo.add_argument("--seed", type=int, default=7)
    return root


def require_new_file(path):
    if Path(path).exists():
        raise FileExistsError(f"Output exists: {path}; choose a new file.")


def split_dataset(data, out, seed):
    scenes = read_scenes(data)
    groups = sorted({s["group_id"] for s in scenes})
    if len(groups) < 10:
        raise ValueError("At least 10 independent group_id values needed for four splits")
    random.Random(seed).shuffle(groups)
    n = len(groups)
    edges = [0, int(n * 0.65), int(n * 0.8), int(n * 0.9), n]
    root = Path(out)
    paths = {key: root / f"{key}.jsonl" for key in ("train", "val", "calibration", "test")}
    for path in paths.values():
        require_new_file(path)
    for i, (key, path) in enumerate(paths.items()):
        selected = set(groups[edges[i] : edges[i + 1]])
        write_scenes(path, [s for s in scenes if s["group_id"] in selected])
    return {key: str(path) for key, path in paths.items()}


def run(args):
    if args.command == "synthetic":
        from .synthetic import generate_dataset

        result = generate_dataset(args.out, args.scenes, args.candidates, args.steps, args.seed)
        return {key: str(path) for key, path in result.items()}
    if args.command == "label":
        from .rules import label_scene

        require_new_file(args.out)
        scenes = read_scenes(args.data)
        if any("labels" in s for s in scenes) and not args.replace_labels:
            raise ValueError("Existing labels: pass --replace-labels to explicitly replace.")
        write_scenes(args.out, [label_scene(s) for s in scenes])
        return {"scenes": len(scenes), "source": "rules_cv", "out": args.out}
    if args.command == "split":
        return split_dataset(args.data, args.out, args.seed)
    if args.command == "train":
        from .training import train_model

        config = json.loads(Path(args.config).read_text()) if args.config else {}
        if args.epochs is not None:
            config["epochs"] = args.epochs
        return train_model(args.train, args.val, args.out, config, args.calibration, args.device)
    if args.command == "evaluate":
        from .inference import evaluate_checkpoint

        require_new_file(args.out)
        result = evaluate_checkpoint(
            args.checkpoint,
            args.data,
            args.device,
            args.allow_seen_groups,
            args.allow_domain_shift,
            args.threshold,
        )
        write_json(args.out, result)
        return result
    if args.command == "rank":
        from .inference import rank_scene

        require_new_file(args.out)
        scenes = read_scenes(args.data)
        if not 0 <= args.scene_index < len(scenes):
            raise ValueError("scene-index out of range")
        result = rank_scene(
            args.checkpoint,
            scenes[args.scene_index],
            args.device,
            args.threshold,
            args.mc_samples,
            args.allow_domain_shift,
            args.max_uncertainty,
        )
        write_json(args.out, result)
        return result
    if args.command == "alpamayo-export":
        from .alpamayo import export_alpamayo

        require_new_file(args.out)
        world = json.loads(Path(args.world_json).read_text()) if args.world_json else None
        scene = export_alpamayo(
            args.clip_id,
            args.t0_us,
            args.out,
            args.samples,
            args.seed,
            world,
            args.model_id,
            args.revision,
            args.ego_length,
            args.ego_width,
        )
        return {
            "scene_id": scene["scene_id"],
            "candidates": len(scene["candidates"]),
            "observation_complete": scene["observation_complete"],
            "out": args.out,
            "notice": "No world sidecar = unlabeled ego-only data; full ranking abstains.",
        }
    if args.command == "demo":
        from .synthetic import generate_dataset
        from .training import train_model
        from .inference import evaluate_checkpoint, rank_scene

        root = Path(args.out)
        if root.exists():
            raise FileExistsError(f"Demo directory exists: {root}; choose a new directory.")
        paths = generate_dataset(root / "data", scenes=args.scenes, seed=args.seed)
        summary = train_model(
            paths["train"],
            paths["val"],
            root / "model",
            {"epochs": args.epochs, "seed": args.seed},
            paths["calibration"],
        )
        report = evaluate_checkpoint(root / "model" / "best.pt", paths["test"])
        ranked = rank_scene(root / "model" / "best.pt", read_scenes(paths["test"])[0])
        write_json(root / "evaluation.json", report)
        write_json(root / "ranking.json", ranked)
        return {"training": summary, "evaluation": report, "output_directory": str(root)}


def main():
    root = parser()
    args = root.parse_args()
    try:
        result = run(args)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        root.exit(2, f"error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
