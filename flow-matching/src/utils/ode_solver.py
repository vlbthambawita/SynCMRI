# src/utils/ode_solver.py
from __future__ import annotations
import torch
from src.models.flowmatching import MergedModel
from flow_matching.solver import ODESolver

def sample_with_solver(
    model:MergedModel,
    x_init,
    solver_config:dict,
    cond=None,
    masks=None,
):
    """
    Uses ODESolver (flow-matching) to sample from x_init -> final output.
    solver_config might contain keys:
        {
          "method": "midpoint"/"rk4"/etc.,
          "step_size": float,
          "time_points": int,
        }

    Returns either the full trajectory [time_points, B, C, H, W] if return_intermediates=True
    or just the final state [B, C, H, W].
    """
    solver = ODESolver(velocity_model=model)

    time_points = solver_config.get("time_points", 10)
    T = torch.linspace(0, 1, time_points, device=x_init.device)

    method = solver_config.get("method", "midpoint")
    step_size = solver_config.get("step_size", 0.02)

    sol = solver.sample(
        time_grid=T,
        x_init=x_init,
        method=method,
        step_size=step_size,
        return_intermediates=True,
        cond=cond,
        masks=masks,
    )
    return sol