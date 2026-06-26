"""Generate class-conditional image samples from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from un0.apple import configure_apple_runtime, generate_samples
from un0.common import resolve_device, save_sample_grid, seed_everything
from un0.data import NUM_CLASSES
from un0.model import (
    PRETRAINED_NAMES,
    ConditionalImplicitKuramotoGenerator,
    build_cifar10_model,
    build_from_config,
    build_imagenet64_model,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Path to a local .pt checkpoint.")
    source.add_argument(
        "--pretrained",
        choices=PRETRAINED_NAMES,
        help="Load released weights from Hugging Face by name.",
    )
    parser.add_argument(
        "--family",
        choices=("cifar10", "imagenet64"),
        default="cifar10",
        help=(
            "Model family for a local --checkpoint (default: cifar10). "
            "Ignored with --pretrained, whose family is fixed by name."
        ),
    )
    parser.add_argument("--output", default="samples/grid.png")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(range(NUM_CLASSES)),
        help="Class ids to sample from (default: all 10).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="How many images to generate per class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="Device for generation (default: auto — CUDA, then MPS, then CPU).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Inference micro-batch size (0 = auto-tune for the loaded model).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override ODE integration steps (lower = faster, may reduce quality).",
    )
    parser.add_argument(
        "--solver",
        choices=("euler", "rk4"),
        default=None,
        help="Override ODE solver (euler is faster on Apple Silicon).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the compile/warmup pass before generating.",
    )
    return parser


def generate(args: argparse.Namespace) -> None:
    """Generate and save samples."""
    configure_apple_runtime()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    if args.pretrained is not None:
        model = ConditionalImplicitKuramotoGenerator.from_pretrained(args.pretrained, device=device)
    else:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        build_fn = build_cifar10_model if args.family == "cifar10" else build_imagenet64_model
        model = build_from_config(build_fn, state.get("config") or {})
        model.load_state_dict(state["model"])
        model = model.to(device)

    classes = torch.tensor(args.classes, device=device, dtype=torch.long)
    class_ids = classes.repeat_interleave(int(args.samples_per_class))
    samples = generate_samples(
        model,
        class_ids,
        device,
        batch_size=int(args.batch_size),
        warmup=not args.no_warmup,
        num_steps=args.num_steps,
        solver=args.solver,
    )
    image_size = round((samples.shape[1] // 3) ** 0.5)
    save_sample_grid(
        samples,
        Path(args.output),
        image_size=image_size,
        nrow=int(args.samples_per_class),
    )


def main() -> None:
    parser = build_parser()
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
