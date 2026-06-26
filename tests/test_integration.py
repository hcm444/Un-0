"""Tests for fixed-step ODE integration helpers."""

from __future__ import annotations

import pytest
import torch

from un0.integration import integrate_fixed, use_compiled_integration
from un0.model import ConditionalKuramotoDynamics


def _tiny_dynamics() -> ConditionalKuramotoDynamics:
    return ConditionalKuramotoDynamics(
        n_oscillators=4,
        n_conditional_oscillators=2,
        num_classes=3,
    )


def test_use_compiled_integration_disabled_while_training() -> None:
    assert use_compiled_integration(torch.device("mps"), training=True) is False
    assert use_compiled_integration(torch.device("mps"), training=False) is True
    assert use_compiled_integration(torch.device("cpu"), training=False) is False


def test_integrate_fixed_runs_for_euler_and_rk4() -> None:
    dynamics = _tiny_dynamics()
    state = torch.randn(2, dynamics.state_dim)
    drive = dynamics.K_drive[torch.tensor([0, 1])]
    step_size = torch.tensor(0.25)

    for solver in ("euler", "rk4"):
        out = integrate_fixed(
            dynamics,
            state,
            drive,
            step_size=step_size,
            num_steps=4,
            solver=solver,  # type: ignore[arg-type]
        )
        assert out.shape == state.shape


def test_coupling_cache_reused_in_eval_mode() -> None:
    dynamics = _tiny_dynamics()
    dynamics.eval()
    state = torch.randn(1, dynamics.state_dim)
    drive = dynamics.K_drive[torch.tensor([0])]

    dynamics(state, state.new_zeros(()), drive)
    first = dynamics._K_eff
    dynamics(state, state.new_zeros(()), drive)
    second = dynamics._K_eff
    assert first is second

    dynamics.train()
    dynamics(state, state.new_zeros(()), drive)
    assert dynamics._coupling_cache_valid is False
