"""Benchmark Un-0 inference on the local accelerator."""

from __future__ import annotations

import argparse
import time

import torch

from un0 import ConditionalImplicitKuramotoGenerator
from un0.apple import (
    autotune_inference_batch_size,
    configure_apple_runtime,
    generate_samples,
    prepare_model_for_inference,
    synchronize_device,
)
from un0.common import resolve_device, seed_everything
from un0.model import PRETRAINED_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained",
        default="cifar10/n4096",
        choices=PRETRAINED_NAMES,
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--presets",
        nargs="*",
        default=["quality", "fast"],
        choices=["quality", "balanced", "fast"],
    )
    return parser


def _timed_generate(
    model: ConditionalImplicitKuramotoGenerator,
    class_ids: torch.Tensor,
    device: torch.device,
    *,
    preset: str,
) -> float:
    synchronize_device(device)
    start = time.perf_counter()
    generate_samples(
        model,
        class_ids,
        device,
        preset=preset,  # type: ignore[arg-type]
        warmup=True,
        autotune_batch=True,
    )
    synchronize_device(device)
    return time.perf_counter() - start


def main() -> None:
    args = build_parser().parse_args()
    configure_apple_runtime()
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))

    print(f"Device: {device}")
    print(f"Loading {args.pretrained}...")
    model = ConditionalImplicitKuramotoGenerator.from_pretrained(args.pretrained, device=device)
    prepare_model_for_inference(model)

    tuned = autotune_inference_batch_size(model, device)
    print(f"Autotuned batch size: {tuned}")
    print(f"Integration: {model.num_steps} {model.solver} steps/solver (checkpoint default)")
    print()

    num_images = int(args.num_images)
    class_ids = (torch.arange(num_images, device=device, dtype=torch.long) % model.dynamics.num_classes)

    for preset in args.presets:
        run_model = ConditionalImplicitKuramotoGenerator.from_pretrained(args.pretrained, device=device)
        elapsed = _timed_generate(run_model, class_ids, device, preset=preset)
        ms_per = (elapsed / num_images) * 1000.0
        print(
            f"{preset:8s}  {num_images:3d} images in {elapsed * 1000:6.0f} ms  "
            f"({ms_per:5.1f} ms/image)  steps={run_model.num_steps} solver={run_model.solver}"
        )


if __name__ == "__main__":
    main()
