"""Explicit, serializable experiment contract; SI units throughout."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

ASSET_CONTRACT_KEYS = (
    "asset_variant", "arm_gravity_compensation", "joint_frictionloss_scale", "joint_armature_scale",
    "wrist_compliance_z_stiffness_n_per_m", "wrist_compliance_z_damping_ns_per_m",
    "wrist_compliance_z_travel_m", "wrist_compliance_xy_stiffness_n_per_m",
    "wrist_compliance_xy_damping_ns_per_m", "wrist_compliance_xy_travel_m",
)


@dataclass
class BlindConfig:
    method: str = "legacy"
    # Version 1 preserves the 2026-10-20 frozen study contract.
    study_revision: int = 1
    assembly_shapes: list[str] = field(default_factory=lambda: ["circle"])
    polygon_peg_width_m: float = 0.0346
    clearance_per_side_m: float = 0.0005
    physics_dt: float = 0.001
    decimation: int = 20
    history_steps: int = 30
    contact_points: int = 64
    # Existing joint inertia, viscous damping, and dry friction are retained.
    kp: list[float] = field(default_factory=lambda: [1500, 2000, 1500, 400, 500, 250])
    kd: list[float] = field(default_factory=lambda: [150, 200, 100, 35, 45, 25])
    torque_limits: list[float] = field(default_factory=lambda: [150, 200, 130, 45, 55, 30])
    action_scale: float = 0.008
    action_transform: str = "identity"
    friction_compensation: bool = True
    friction_deadband_rad: float = 0.0005
    goal_depth_m: float = 0.04
    milestone_depths_m: list[float] = field(default_factory=lambda: [
        0., .00025, .0005, .001, .002, .003, .004, .005, .01, .02, .04])
    milestone_rewards: list[float] = field(default_factory=lambda: [
        .2, .25, .25, .5, .5, .5, .5, 2., 3., 4., 8.])
    success_hold_s: float = 0.5
    success_speed_mps: float = 0.01
    geometry_tolerance_m: float = 0.000025
    horizon_s: float = 8.0
    initial_gap_m: float = 0.0015
    xy_min_m: float = 0.0
    xy_max_m: float = 0.00025
    tilt_deg: float = 0.2
    world_xy_jitter_m: float = 0.002
    force_abort_n: float = 150.0
    contact_band_m: float = 0.001
    seed: int = 7
    num_envs: int = 64
    rollout_steps: int = 32
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 0.0003
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.003
    contact_coef: float = 1.0
    contact_weights: list[float] = field(default_factory=lambda: [0.25, 3.0, 2.0])
    max_grad_norm: float = 1.0
    physics_backend: str = "contact-fix"
    reset_bank_size: int = 128
    fixture_xy_jitter_m: float = 0.05
    fixture_z_jitter_m: float = 0.02
    fixture_tilt_deg: float = 5.0
    reset_xy_halfwidth_m: float = 0.0125
    plate_friction_range: list[float] = field(default_factory=lambda: [0.5, 2.0])
    plate_timeconst_range_s: list[float] = field(default_factory=lambda: [0.0035, 0.0071])
    velocity_scale: list[float] = field(default_factory=lambda: [2., 2., 2., 3., 3., 3.])
    seating_tolerance_m: float = 0.001
    capture_depth_m: float = 0.001
    far_distance_m: float = 0.25
    action_clip: float | None = 1.0
    nconmax: int = 512
    njmax: int = 2048
    checkpoint_interval: int = 50
    # Long-run launcher schedule; the low-level train CLI still takes --iterations.
    job_iterations: int | None = None
    job_evaluation_iterations: list[int] = field(default_factory=list)
    profile_start_iteration: int = -1
    profile_stop_iteration: int = -1

    # Asset selection is independent of the frozen revision-4 learning task.
    asset_variant: str = "rigid"
    arm_gravity_compensation: bool = False
    joint_frictionloss_scale: float = 1.0
    joint_armature_scale: float = 1.0
    wrist_compliance_z_stiffness_n_per_m: float = 4000.0
    wrist_compliance_z_damping_ns_per_m: float = 60.0
    wrist_compliance_z_travel_m: float = 0.004
    wrist_compliance_xy_stiffness_n_per_m: float = 6000.0
    wrist_compliance_xy_damping_ns_per_m: float = 60.0
    wrist_compliance_xy_travel_m: float = 0.002

    # Revision 6: tool-frame Cartesian tip action, tip F/T observation, bounded
    # force governor, chamfer-consistent seating and dense in-hole shaping.
    # Ignored by revisions 1-4.
    cart_translation_step_m: float = 0.006
    cart_rotation_step_rad: float = 0.006
    cart_translation_lead_m: float = 0.012
    cart_rotation_lead_rad: float = 0.012
    cart_dls_damping: float = 0.01
    cart_rotation_weight_m: float = 0.1
    lateral_force_threshold_n: float = 10.0
    axial_force_threshold_n: float = 20.0
    force_band_n: float = 5.0
    force_retreat_m_per_ns: float = 0.002
    moment_threshold_nm: float = 0.3
    moment_band_nm: float = 0.5
    moment_comply_rad_per_nms: float = 0.05
    governor_wrench_tau_s: float = 0.005
    wrench_force_scale_n: float = 50.0
    wrench_moment_scale_nm: float = 2.0
    preload_force_range_n: list[float] = field(default_factory=lambda: [5.0, 25.0])
    preload_bonus: float = 0.0
    depth_progress_reward: float = 1.0
    tilt_progress_reward_per_rad: float = 5.0
    yaw_progress_reward_per_rad: float = 2.0
    overload_force_n: float = 35.0
    overload_moment_nm: float = 1.5
    overload_force_weight: float = 0.0001
    overload_moment_weight: float = 0.01
    overload_penalty_cap: float = 0.05
    pose_coef: float = 0.05
    moment_abort_nm: float = 2.0
    load_fault_penalty: float = 0.0
    # Explicit evaluation-only ablation for an existing revision-4 checkpoint.
    seating_protocol: str = "legacy_v4"

    @property
    def wrist_compliance_enabled(self):
        return self.asset_variant == "passive_xyz_v1"

    def validate(self):
        import math
        if self.seating_protocol not in ("legacy_v4", "chamfer_v1"):
            raise ValueError("unknown seating protocol")
        if self.seating_protocol != "legacy_v4" and self.study_revision not in (4, 6):
            raise ValueError("chamfer seating requires revision 4 or 6")
        if self.asset_variant not in ("rigid", "passive_xyz_v1"):
            raise ValueError("unknown asset variant")
        if self.wrist_compliance_enabled and (self.method != "study" or self.study_revision not in (4, 6)):
            raise ValueError("passive assets require the revision 4 or 6 study task")
        if self.study_revision == 6:
            if not self.wrist_compliance_enabled:
                raise ValueError("revision 6 requires the passive wrist asset (nominal tip site)")
            if self.action_transform != "tanh":
                raise ValueError("revision 6 maps actions with tanh")
            positive = ("cart_translation_step_m", "cart_rotation_step_rad", "cart_translation_lead_m",
                        "cart_rotation_lead_rad", "cart_dls_damping", "cart_rotation_weight_m",
                        "lateral_force_threshold_n", "axial_force_threshold_n", "force_band_n",
                        "moment_threshold_nm", "moment_band_nm", "governor_wrench_tau_s",
                        "wrench_force_scale_n", "wrench_moment_scale_nm", "overload_force_n",
                        "overload_moment_nm", "force_abort_n", "moment_abort_nm")
            for key in positive:
                if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                    raise ValueError(f"revision 6 parameter must be positive: {key}")
            if self.cart_translation_step_m > self.cart_translation_lead_m:
                raise ValueError("one policy step may not exceed the translation lead bound")
            if self.cart_rotation_step_rad > self.cart_rotation_lead_rad:
                raise ValueError("one policy step may not exceed the rotation lead bound")
            lo, hi = self.preload_force_range_n
            if not 0 <= lo < hi:
                raise ValueError("invalid preload force range")
            if self.seating_protocol != "chamfer_v1":
                raise ValueError("revision 6 requires explicit chamfer_v1 seating")
            if (self.force_abort_n <= max(self.lateral_force_threshold_n, self.axial_force_threshold_n)
                    or self.moment_abort_nm <= self.moment_threshold_nm):
                raise ValueError("abort limits must exceed the soft governor thresholds")
            for key in ("load_fault_penalty", "force_retreat_m_per_ns", "moment_comply_rad_per_nms",
                        "preload_bonus", "depth_progress_reward", "tilt_progress_reward_per_rad",
                        "yaw_progress_reward_per_rad", "overload_force_weight", "overload_moment_weight",
                        "overload_penalty_cap", "pose_coef"):
                if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                    raise ValueError(f"revision 6 parameter must be nonnegative: {key}")
        if self.asset_variant == "rigid" and (self.arm_gravity_compensation or
                self.joint_frictionloss_scale != 1.0 or self.joint_armature_scale != 1.0):
            raise ValueError("modified joint dynamics require an explicit asset variant")
        if (not math.isfinite(self.joint_frictionloss_scale) or not 0 <= self.joint_frictionloss_scale <= 1
                or not math.isfinite(self.joint_armature_scale) or not 0 < self.joint_armature_scale <= 1):
            raise ValueError("invalid asset joint dynamics")
        for key in ASSET_CONTRACT_KEYS[4:]:
            if not math.isfinite(getattr(self,key)) or getattr(self,key) <= 0:
                raise ValueError(f"invalid asset parameter: {key}")
        if max(self.wrist_compliance_z_travel_m,self.wrist_compliance_xy_travel_m) > .01:
            raise ValueError("passive travel must not exceed 10 mm")
        if self.job_iterations is not None and (type(self.job_iterations) is not int or self.job_iterations <= 0):
            raise ValueError("job_iterations must be a positive integer or null")
        if (any(type(n) is not int or n <= 0 for n in self.job_evaluation_iterations)
            or (self.job_evaluation_iterations and self.job_iterations is None)
            or any(n > self.job_iterations for n in self.job_evaluation_iterations)
            or sorted(set(self.job_evaluation_iterations)) != self.job_evaluation_iterations):
            raise ValueError("job evaluation iterations must be increasing and within the job budget")
        if self.method not in ("legacy", "study"):
            raise ValueError("unknown method")
        if self.study_revision not in (1, 2, 3, 4, 6):
            raise ValueError("unknown study revision")
        if self.study_revision >= 3:
            allowed = ("square", "hexagon") if self.study_revision == 3 else ("circle", "square", "hexagon")
            if self.method != "study" or not self.assembly_shapes or any(
                s not in allowed for s in self.assembly_shapes
            ) or len(set(self.assembly_shapes)) != len(self.assembly_shapes):
                raise ValueError(f"revision {self.study_revision} requires distinct shapes from {allowed}")
            if self.num_envs % len(self.assembly_shapes) or self.reset_bank_size % len(self.assembly_shapes):
                raise ValueError("mixed shapes require equal environment and reset-bank counts")
            if not .02 <= self.polygon_peg_width_m <= .04:
                raise ValueError("unsupported polygon across-flats width")
        elif self.assembly_shapes != ["circle"]:
            raise ValueError("mixed assemblies require study revision 3 or 4")
        if self.action_transform not in ("identity", "tanh"):
            raise ValueError("unknown action transform")
        if not 0 < self.clearance_per_side_m < 0.002:
            raise ValueError("clearance_per_side_m is a radial clearance in metres")
        if not 0 < self.goal_depth_m <= 0.04:
            raise ValueError("audited depth is at most the 40 mm plate thickness")
        if not 0 <= self.xy_min_m <= self.xy_max_m <= 0.01:
            raise ValueError("invalid reset radius")
        if self.physics_dt <= 0 or self.decimation < 1 or self.history_steps != 30:
            raise ValueError("invalid timing/history")
        if self.num_envs < 1 or self.contact_points != 64:
            raise ValueError("invalid environment/label count")
        if not -.005 <= self.initial_gap_m <= .01 or -self.initial_gap_m >= self.goal_depth_m:
            raise ValueError("reset must be before the goal depth")
        if self.initial_gap_m < 0 and (self.xy_max_m>.00025 or self.tilt_deg>.2):
            raise ValueError("shallow captured curriculum is limited to audited aligned resets")
        if self.action_scale<=0 or self.friction_deadband_rad<=0 or self.success_hold_s<0:
            raise ValueError("invalid controller or hold parameters")
        if self.method == "legacy" and self.success_hold_s == 0:
            raise ValueError("legacy protocol requires positive hold")
        if self.reset_bank_size < 1 or self.minibatches > self.num_envs*self.rollout_steps:
            raise ValueError("invalid reset bank or PPO batch")
        for name in ("plate_friction_range", "plate_timeconst_range_s"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(f"invalid {name}")
        if self.action_clip is not None and self.action_clip <= 0:
            raise ValueError("action clip must be positive or null")
        if self.method == "study" and (
            self.goal_depth_m <= self.seating_tolerance_m
            or not 0 < self.capture_depth_m < self.goal_depth_m-self.seating_tolerance_m
            or self.reset_xy_halfwidth_m <= self.clearance_per_side_m
        ):
            raise ValueError("invalid study task geometry")
        if (len(self.milestone_depths_m)!=len(self.milestone_rewards)
            or sorted(set(self.milestone_depths_m))!=self.milestone_depths_m):
            raise ValueError("milestones must be unique, increasing and match rewards")
        for name in ("kp", "kd", "torque_limits"):
            values = getattr(self, name)
            if len(values) != 6 or min(values) <= 0:
                raise ValueError(f"{name} must contain six positive values")
        return self

    @property
    def shape_vocabulary(self):
        # Stable critic columns for revision 4, including single-shape probes
        # or future leave-one-shape-out experiments. Never part of actor input.
        return ("circle", "square", "hexagon") if self.study_revision in (4, 6) else tuple(self.assembly_shapes)

    @property
    def policy_dt(self):
        return self.physics_dt * self.decimation

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None, **overrides):
        data = {} if path is None else json.loads(Path(path).read_text())
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data).validate()
