"""MJLab asset adapter for project-local Estun and peg-hole models."""

from __future__ import annotations

from pathlib import Path

import mujoco
from functools import partial
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets"
START_ARM_QPOS = (0.052305, 0.858200, 0.229875, -0.018345, 0.478656, -0.026586)
# Calibrated by applying START_ARM_QPOS and calling mj_forward. Reset events
# run before MuJoCo refreshes site_xpos, so they must not read that stale data.
START_PEG_TIP_WORLD = (
    0.8659840917335416,
    -0.001690045032524172,
    0.10003137869487111,
)
# Physical centre of the pa2 cylinder's front face at START_ARM_QPOS.  The
# source control site above is deliberately offset by 0.375 mm in hand X.
START_PEG_GEOMETRY_TIP_WORLD = (
    0.8663573919923548,
    -0.0017256809690146674,
    0.10003279177677973,
)
# Radius of the pa2 collision cylinder measured from the collision OBJ in the
# step-1 geometry audit.  The front face is not perfectly horizontal at the
# calibrated start pose, so its world-XY slope is retained as part of the safe
# reset clearance calculation.
PEG_COLLISION_RADIUS = 34.899e-3 / 2.0
START_PEG_FACE_SLOPE_XY = (
    -0.004570231951331698,
    -0.00822166864461721,
)
# Visual test: tip center 5 mm above the mouth, 5 degrees about world X.
ATTITUDE_TEST_ARM_QPOS = (0.08491493441004788, 0.8562043876039094, 0.22604012528562303, -0.18116760365099444, 0.5029094854689814, 0.14921928100452286)
# Preserve the source XML placement, including its original lateral offset.
NOMINAL_HOLE_MOUTH_WORLD = (0.85, 0.0, 0.1)


def asset_root() -> Path:
    """Return the self-contained model directory shipped with this project."""
    return ASSET_ROOT


def detailed_assembly_spec() -> mujoco.MjSpec:
    """Load the robot, 34.9 mm peg, and detailed 35 mm hole as one entity."""
    scene_path = asset_root() / "scene.xml"
    if not scene_path.is_file():
        raise FileNotFoundError(
            f"Project scene not found: {scene_path}."
        )
    return mujoco.MjSpec.from_file(str(scene_path))


def detailed_assembly_cfg() -> EntityCfg:
    """Return an MJLab entity configuration backed by the original STL scene."""
    return EntityCfg(
        spec_fn=detailed_assembly_spec,
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={"joint_[1-6]": 0.0, ".*driver_joint": 0.0},
            joint_vel={".*": 0.0},
        ),
    )


def robot_spec() -> mujoco.MjSpec:
    """Load the Estun robot with its attached 34.9 mm peg."""
    path = asset_root() / "robot" / "robot.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Estun robot not found: {path}")
    spec = mujoco.MjSpec.from_file(str(path))
    # Evaluation aids: the cyan sphere is the controlled peg-front center and
    # the cyan cylinder makes the peg axis visible at close range.
    tip = spec.site("peg_tip_site")
    tip.type = mujoco.mjtGeom.mjGEOM_SPHERE
    tip.size[:] = (0.003, 0.003, 0.003)
    tip.rgba[:] = (0.0, 0.9, 1.0, 1.0)
    # The source control/pivot site above is intentionally preserved.  Its X
    # coordinate is 0.375 mm away from the pa2 cylinder centre, so geometry
    # labels and success checks need a separate site on the physical peg axis.
    spec.body("hand_mount").add_site(
        name="peg_geometry_tip_site",
        pos=(0.019625, 0.004625, 0.238),
        size=(0.0005, 0.0005, 0.0005),
        rgba=(0.0, 0.0, 0.0, 0.0),
    )
    spec.body("hand_mount").add_site(
        name="peg_axis_visual",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        # The front center is at z=0.238 and the shaft extends back toward
        # the wrist (-Z).  Centering the marker at the tip incorrectly drew
        # half of it beyond the pivot and made the rotation look off-center.
        pos=(0.019625, 0.004625, 0.183),
        size=(0.0012, 0.055, 0.0),
        rgba=(0.0, 0.9, 1.0, 0.75),
    )
    return spec


def detailed_hole_spec(collision_model='legacy') -> mujoco.MjSpec:
    """Load the detailed hole and move its placement into the MJLab entity pose."""
    path = asset_root() / "hole" / "Hole35.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Detailed hole not found: {path}")
    spec = mujoco.MjSpec.from_file(str(path))
    if collision_model=='partitioned':
        from .hole_collision import replace_collision
        replace_collision(spec)
    elif collision_model!='legacy':
        raise ValueError(collision_model)
    root = spec.body("root_hole")
    # The source root contains world placement (0.85, 0, 0.1).  MJLab must own
    # this transform so each environment can randomize its mocap pose safely.
    root.pos[:] = (0.0, 0.0, 0.0)
    root.add_site(
        name="hole_mouth_site",
        pos=(0.0, 0.0, 0.0),
        size=(0.002, 0.002, 0.002),
        rgba=(1.0, 0.1, 0.8, 1.0),
    )
    # Magenta cylinder is the true hole center axis.  It is visualization-only.
    root.add_site(
        name="hole_axis_visual",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.055),
        size=(0.0012, 0.055, 0.0),
        rgba=(1.0, 0.1, 0.8, 0.75),
    )
    return spec


def robot_cfg(initial_arm_qpos: tuple[float, ...] = START_ARM_QPOS) -> EntityCfg:
    """MJLab robot entity retaining the position actuators from robot.xml."""
    arm_actuators = tuple(
        XmlActuatorCfg(
            target_names_expr=(f"joint_{index}",), command_field="position"
        )
        for index in range(1, 7)
    )
    gripper_actuator = XmlActuatorCfg(
        target_names_expr=("split",),
        command_field="position",
        transmission_type=TransmissionType.TENDON,
    )
    return EntityCfg(
        spec_fn=robot_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=arm_actuators + (gripper_actuator,),
            soft_joint_pos_limit_factor=0.95,
        ),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={
                **{
                    f"joint_{index}": position
                    for index, position in enumerate(initial_arm_qpos, start=1)
                },
                ".*driver_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )


def detailed_hole_cfg(collision_model='legacy') -> EntityCfg:
    """Detailed fixed hole represented as a resettable MJLab mocap entity."""
    return EntityCfg(
        spec_fn=partial(detailed_hole_spec,collision_model),
        init_state=EntityCfg.InitialStateCfg(
            pos=NOMINAL_HOLE_MOUTH_WORLD,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
