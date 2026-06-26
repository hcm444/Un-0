"""Tests for shared helpers in un0.common."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from un0.common import best_available_device, resolve_device


def test_resolve_device_explicit_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_prefers_cuda() -> None:
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.backends.mps, "is_available", return_value=True),
    ):
        assert resolve_device("auto") == torch.device("cuda")


def test_resolve_device_auto_falls_back_to_mps() -> None:
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.backends.mps, "is_available", return_value=True),
    ):
        assert resolve_device("auto") == torch.device("mps")


def test_resolve_device_auto_falls_back_to_cpu() -> None:
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.backends.mps, "is_available", return_value=False),
    ):
        assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_mps_unavailable_raises() -> None:
    with (
        patch.object(torch.backends.mps, "is_available", return_value=False),
        pytest.raises(RuntimeError, match="MPS requested"),
    ):
        resolve_device("mps")


def test_resolve_device_cuda_unavailable_raises() -> None:
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        pytest.raises(RuntimeError, match="CUDA requested"),
    ):
        resolve_device("cuda")


def test_best_available_device_matches_auto() -> None:
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.backends.mps, "is_available", return_value=True),
    ):
        assert best_available_device() == resolve_device("auto")
