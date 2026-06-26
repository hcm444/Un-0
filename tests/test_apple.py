"""Tests for Apple Silicon inference helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from un0.apple import (
    apply_inference_preset,
    default_inference_batch_size,
    default_pretrained_checkpoint,
    generate_samples,
    sample_batched,
)


def _mock_model(*, n_oscillators: int = 1024, num_classes: int = 10) -> MagicMock:
    model = MagicMock()
    model.dynamics.n = n_oscillators
    model.dynamics.num_classes = num_classes
    model.num_steps = 25
    model.solver = "rk4"
    model.default_num_steps = 25
    model.default_solver = "rk4"
    model.sample.side_effect = lambda ids: torch.zeros(ids.shape[0], 3072)
    return model


def test_default_inference_batch_size_scales_with_model() -> None:
    assert default_inference_batch_size(_mock_model(n_oscillators=1024)) == 32
    assert default_inference_batch_size(_mock_model(n_oscillators=4096)) == 64
    assert default_inference_batch_size(_mock_model(n_oscillators=16384)) == 8


def test_default_pretrained_checkpoint_prefers_quality_on_mps() -> None:
    with patch("un0.apple.is_apple_silicon", return_value=True):
        assert default_pretrained_checkpoint() == "cifar10/n4096"
    with patch("un0.apple.is_apple_silicon", return_value=False):
        assert default_pretrained_checkpoint() == "cifar10/n1024"


def test_sample_batched_concatenates_chunks() -> None:
    model = _mock_model()
    model.sample.side_effect = lambda ids: torch.full((ids.shape[0], 3072), float(ids[0].item()))
    out = sample_batched(model, torch.tensor([0, 1, 2, 3, 4]), batch_size=2)
    assert out.shape == (5, 3072)
    assert model.sample.call_count == 3


def test_sample_batched_rejects_empty_ids() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        sample_batched(_mock_model(), torch.tensor([]), batch_size=2)


def test_generate_samples_applies_fast_overrides() -> None:
    model = _mock_model()
    device = torch.device("cpu")
    ids = torch.tensor([0, 1])
    generate_samples(
        model,
        ids,
        device,
        batch_size=2,
        warmup=False,
        preset="fast",
    )
    assert model.num_steps == 10
    assert model.solver == "euler"
    model.sample.assert_called_once()


def test_apply_inference_preset_balanced() -> None:
    model = _mock_model()
    model.default_num_steps = 25
    model.default_solver = "rk4"
    model.num_steps = 25
    model.solver = "rk4"
    apply_inference_preset(model, "balanced")
    assert model.num_steps == 15
    assert model.solver == "rk4"
    apply_inference_preset(model, "quality")
    assert model.num_steps == 25
    assert model.solver == "rk4"
