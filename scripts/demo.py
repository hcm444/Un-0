"""Generate a quick preview grid from released Un-0 weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from un0 import ConditionalImplicitKuramotoGenerator
from un0.apple import (
    configure_apple_runtime,
    default_pretrained_checkpoint,
    generate_samples,
    is_apple_silicon,
)
from un0.common import resolve_device, seed_everything
from un0.model import PRETRAINED_NAMES

CIFAR10_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained",
        default=None,
        choices=PRETRAINED_NAMES,
        help="Released checkpoint (default: n4096 on Apple Silicon, else n1024).",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Class ids to sample.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1,
        help="Images per class.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("samples/demo.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="Device for generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Inference micro-batch size (0 = auto-tune for the loaded model).",
    )
    parser.add_argument(
        "--preset",
        choices=("quality", "balanced", "fast"),
        default="quality",
        help="Inference quality/speed preset (default: quality).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut for --preset fast.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the compile/warmup pass before generating.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_apple_runtime()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))

    pretrained = args.pretrained or default_pretrained_checkpoint()
    print(f"Loading {pretrained} on {device}...")
    model = ConditionalImplicitKuramotoGenerator.from_pretrained(pretrained, device=device)

    class_ids = torch.tensor(args.classes, device=device, dtype=torch.long).repeat_interleave(
        int(args.samples_per_class)
    )
    preset = "fast" if args.fast else str(args.preset)
    flat = generate_samples(
        model,
        class_ids,
        device,
        batch_size=int(args.batch_size),
        warmup=not args.no_warmup,
        preset=preset,  # type: ignore[arg-type]
    )
    size = round((flat.shape[1] // 3) ** 0.5)
    images = flat.reshape(-1, 3, size, size)
    images = ((images + 1.0) * 0.5).clamp(0.0, 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, args.output, nrow=int(args.samples_per_class))

    labels = [
        CIFAR10_LABELS[c] if pretrained.startswith("cifar10/") else str(c)
        for c in args.classes
    ]
    apple_note = " (Apple Silicon optimizations on)" if is_apple_silicon() and device.type == "mps" else ""
    print(f"Wrote {args.output} ({len(class_ids)} images, classes: {', '.join(labels)}){apple_note}")


if __name__ == "__main__":
    main()
