"""Apple Silicon inference helpers for faster MPS image generation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from un0.model import ConditionalImplicitKuramotoGenerator

InferencePreset = Literal["quality", "balanced", "fast"]

_PRESET_OVERRIDES: dict[InferencePreset, dict[str, int | str | None]] = {
    "quality": {},
    "balanced": {"num_steps": 15, "solver": "rk4"},
    "fast": {"num_steps": 10, "solver": "euler"},
}


def is_apple_silicon() -> bool:
    """Return True when PyTorch can run kernels on the Metal backend."""
    return torch.backends.mps.is_available()


def configure_apple_runtime() -> None:
    """Tune PyTorch for sustained MPS inference workloads."""
    if not is_apple_silicon():
        return
    # Use all available unified memory before paging; default cap can throttle GPU work.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    # Prefer tensor-core friendly fp32 matmuls on Apple Silicon.
    torch.set_float32_matmul_precision("high")


def synchronize_device(device: torch.device | str) -> None:
    """Block until queued work on the accelerator has finished."""
    dev = torch.device(device)
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def default_inference_batch_size(model: ConditionalImplicitKuramotoGenerator) -> int:
    """Pick a batch size that keeps MPS busy without exhausting unified memory."""
    n_oscillators = int(model.dynamics.n)
    if n_oscillators >= 16_384:
        return 8
    if n_oscillators >= 10_240:
        return 16
    if n_oscillators >= 4096:
        return 64
    return 32


def apply_inference_preset(
    model: ConditionalImplicitKuramotoGenerator,
    preset: InferencePreset,
) -> None:
    """Apply a quality/speed preset by overriding solver integration settings."""
    if preset == "quality":
        model.num_steps = int(model.default_num_steps)
        model.solver = model.default_solver
        return

    overrides = _PRESET_OVERRIDES[preset]
    if overrides["num_steps"] is not None:
        model.num_steps = int(overrides["num_steps"])
    if overrides["solver"] is not None:
        model.solver = str(overrides["solver"])  # type: ignore[assignment]


def default_pretrained_checkpoint() -> str:
    """Prefer the best CIFAR checkpoint when MPS can run it at interactive speed."""
    return "cifar10/n4096" if is_apple_silicon() else "cifar10/n1024"


@torch.inference_mode()
def warmup_inference(
    model: ConditionalImplicitKuramotoGenerator,
    device: torch.device,
    *,
    batch_size: int = 2,
) -> None:
    """Run a tiny forward pass so torch.compile kernels are ready before user-facing work."""
    num_classes = int(model.dynamics.num_classes)
    warmup_ids = torch.arange(min(batch_size, num_classes), device=device, dtype=torch.long)
    model.sample(warmup_ids)
    synchronize_device(device)


@torch.inference_mode()
def sample_batched(
    model: ConditionalImplicitKuramotoGenerator,
    class_ids: Tensor,
    *,
    batch_size: int,
) -> Tensor:
    """Generate samples in fixed-size chunks for better MPS utilization."""
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}."
        raise ValueError(msg)
    if class_ids.numel() == 0:
        msg = "class_ids must not be empty."
        raise ValueError(msg)

    chunks: list[Tensor] = []
    for start in range(0, int(class_ids.shape[0]), batch_size):
        batch_ids = class_ids[start : start + batch_size]
        chunks.append(model.sample(batch_ids))
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def generate_samples(
    model: ConditionalImplicitKuramotoGenerator,
    class_ids: Tensor,
    device: torch.device,
    *,
    batch_size: int = 0,
    warmup: bool = True,
    num_steps: int | None = None,
    solver: str | None = None,
    preset: InferencePreset | None = None,
) -> Tensor:
    """Run class-conditional generation with Apple-friendly defaults."""
    if preset is not None:
        apply_inference_preset(model, preset)
    if num_steps is not None:
        model.num_steps = int(num_steps)
    if solver is not None:
        model.solver = solver  # type: ignore[assignment]

    effective_batch = batch_size or default_inference_batch_size(model)
    if warmup and device.type in ("mps", "cuda"):
        warmup_inference(model, device, batch_size=min(effective_batch, 2))

    return sample_batched(model, class_ids, batch_size=effective_batch)
