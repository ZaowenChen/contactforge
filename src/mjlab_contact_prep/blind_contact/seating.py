"""Chamfer-consistent finite-depth seating, shared by v6 and the v4 ablation."""
ENTRY_CHAMFER_M = .001
EXIT_START_M = -.0395
EXIT_CHAMFER_M = .0005


def chamfer_allowance(z):
    """Extra radial/normal room of the bore at height z (hole frame, z<0 inside)."""
    entry = (z+ENTRY_CHAMFER_M).clamp(0., ENTRY_CHAMFER_M)
    exit_ = (EXIT_START_M-z).clamp(0., EXIT_CHAMFER_M)
    return entry+exit_


def circle_events(points, radius, rotation, shaft_error, cfg):
    height = points[..., 2]
    allowance = chamfer_allowance(height)+cfg.geometry_tolerance_m
    inside = points[..., :2].norm(dim=-1) <= radius+cfg.clearance_per_side_m+allowance
    u, v = points[:, 1, :2]-points[:, 0, :2], points[:, 2, :2]-points[:, 0, :2]
    downward = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0] < 0
    legal = (shaft_error <= cfg.geometry_tolerance_m) & downward & inside.all(-1)
    captured = legal & (height.max(-1).values <= -cfg.capture_depth_m)
    seated = legal & ((height+cfg.goal_depth_m).abs() <= cfg.seating_tolerance_m).all(-1)
    return captured, seated


def polygon_events(geometry, points, rotation, shaft_error, cfg):
    height = points[..., 2]
    allowance = chamfer_allowance(height)+cfg.geometry_tolerance_m
    inside = (points[..., :2]@geometry.normals.T <= geometry.apothem+allowance[..., None]).all(-1)
    legal = (shaft_error <= cfg.geometry_tolerance_m) & (rotation[:, 2, 2] < 0) & inside.all(-1)
    captured = legal & (height.max(-1).values <= -cfg.capture_depth_m)
    seated = legal & ((height+cfg.goal_depth_m).abs() <= cfg.seating_tolerance_m).all(-1)
    return captured, seated

