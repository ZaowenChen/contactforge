"""Batched PyTorch wrench processing for MJLab environments."""

from __future__ import annotations

import torch
from functools import lru_cache


@lru_cache(maxsize=128)
def constant_tensor(value, *, dtype, device):
    """Reuse read-only scalar/tuple constants, keyed by value, dtype and device.

    Callers must not mutate the returned tensor. Changed configuration values
    produce a new entry, so the cache also respects runtime parameter changes.
    """
    return torch.as_tensor(value, dtype=dtype, device=device)


def compensate_payload_gravity(
    raw_wrench_sensor: torch.Tensor,
    rotation_world_from_sensor: torch.Tensor,
    sensor_to_payload_com_sensor: torch.Tensor,
    payload_mass: float | torch.Tensor,
    gravity_world: torch.Tensor,
    *,
    negate_sensor_output: bool = True,
) -> torch.Tensor:
    """Remove the rigid payload's static gravity wrench in sensor axes."""
    measured = -raw_wrench_sensor if negate_sensor_output else raw_wrench_sensor
    mass = (payload_mass.to(dtype=measured.dtype, device=measured.device)
            if torch.is_tensor(payload_mass) else constant_tensor(
                float(payload_mass), dtype=measured.dtype, device=measured.device))
    gravity_force_world = gravity_world * mass
    if gravity_force_world.ndim == 1:
        gravity_force_world = gravity_force_world.expand(measured.shape[0], -1)
    gravity_sensor = torch.bmm(
        rotation_world_from_sensor.transpose(1, 2), gravity_force_world.unsqueeze(-1)
    ).squeeze(-1)
    gravity_moment_sensor = torch.linalg.cross(
        sensor_to_payload_com_sensor, gravity_sensor, dim=-1
    )
    return torch.cat(
        (
            measured[:, :3] - gravity_sensor,
            measured[:, 3:] - gravity_moment_sensor,
        ),
        dim=-1,
    )


def transform_wrench_sensor_to_target_batch(
    wrench_at_sensor: torch.Tensor,
    sensor_to_target_sensor: torch.Tensor,
    rotation_target_from_sensor: torch.Tensor,
) -> torch.Tensor:
    """Shift and rotate batched sensor-origin wrenches to target origins."""
    force_sensor = wrench_at_sensor[:, :3]
    moment_target_sensor = wrench_at_sensor[:, 3:] - torch.linalg.cross(
        sensor_to_target_sensor, force_sensor, dim=-1
    )
    force_target = torch.bmm(
        rotation_target_from_sensor, force_sensor.unsqueeze(-1)
    ).squeeze(-1)
    moment_target = torch.bmm(
        rotation_target_from_sensor, moment_target_sensor.unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat((force_target, moment_target), dim=-1)
