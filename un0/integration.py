"""Fixed-step Kuramoto ODE integration (eager and torch.compile)."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from un0.model import ConditionalKuramotoDynamics

Solver = Literal["euler", "rk4"]


def integrate_fixed(
    dynamics: ConditionalKuramotoDynamics,
    initial_state: Tensor,
    drive: Tensor,
    *,
    step_size: Tensor,
    num_steps: int,
    solver: Solver,
) -> Tensor:
    """Integrate with a fixed-step Euler or RK4 loop, returning the final state only."""
    state = initial_state
    t = initial_state.new_zeros(())
    h = step_size

    if solver == "euler":
        for _ in range(num_steps):
            state = state + h * dynamics(state, t, drive)
            t = t + h
        return state

    for _ in range(num_steps):
        k1 = dynamics(state, t, drive)
        half_h = 0.5 * h
        k2 = dynamics(state + half_h * k1, t + half_h, drive)
        k3 = dynamics(state + half_h * k2, t + half_h, drive)
        k4 = dynamics(state + h * k3, t + h, drive)
        state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + h
    return state


@functools.cache
def _compiled_integrator(solver: Solver, num_steps: int) -> Callable[..., Tensor]:
    """Return a torch.compile-wrapped integrator for a fixed solver and step count."""

    @torch.compile
    def _run(
        dynamics: ConditionalKuramotoDynamics,
        initial_state: Tensor,
        drive: Tensor,
        step_size: Tensor,
    ) -> Tensor:
        return integrate_fixed(
            dynamics,
            initial_state,
            drive,
            step_size=step_size,
            num_steps=num_steps,
            solver=solver,
        )

    return _run


def integrate_fixed_compiled(
    dynamics: ConditionalKuramotoDynamics,
    initial_state: Tensor,
    drive: Tensor,
    *,
    step_size: Tensor,
    num_steps: int,
    solver: Solver,
) -> Tensor:
    """Integrate via a cached torch.compile graph when available."""
    return _compiled_integrator(solver, int(num_steps))(
        dynamics,
        initial_state,
        drive,
        step_size,
    )


def use_compiled_integration(device: torch.device, *, training: bool) -> bool:
    """Return whether compiled integration is safe and worthwhile on this device."""
    if training:
        return False
    return device.type in ("mps", "cuda")
