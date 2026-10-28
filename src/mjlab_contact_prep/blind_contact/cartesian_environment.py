"""Revision 6 task on the passive-wrist assets.

Changes relative to the frozen revision 4 transition (which stays untouched in
``study_environment.py`` / ``polygon_environment.py``):

1. Seating/capture containment of the 64 bottom-rim samples uses the same
   z-dependent entry/exit chamfer allowance and 25 um tolerance as the whole
   shaft legality check. Revision 4 used a zero-allowance straight-bore test,
   which rejected legal 40 mm seats by a few micrometres inside the exit
   chamfer and let the peg run through the open bore.
2. Actions move a tool-frame Cartesian tip reference (``cartesian_control``),
   governed every physics step by the compensated wrist F/T.
3. The actor receives 36 values per frame: the 30 revision-4 encoder values
   plus the interval-mean tip wrench (force / 50 N, moment / 2 N m).
4. Reward keeps first legal capture +1 and success +1, and adds bounded dense
   terms: legal depth progress, tilt/yaw potential, a small preload bonus
   before capture and a squared overload penalty from 1 kHz peaks.
5. Observations carry a privileged 6-D pose-error target for an auxiliary
   regression head (training only; never an actor input).
"""
from dataclasses import replace
import math
import mujoco
import torch
from .cartesian_control import CartesianTipImpedance
from .passive_assets import passive_indices
from .polygon_environment import CircleStudyEnv, PolygonStudyEnv, MixedStudyEnv, _ReadOnlyData
from .study_environment import StudyInsertionEnv
from ..wrench_torch import compensate_payload_gravity, transform_wrench_sensor_to_target_batch

from .seating import chamfer_allowance, circle_events, polygon_events
from .cartesian_control import so3_log

FRAME_DIM = 36

def alignment_potential(tilt, yaw, captured, cfg):
    cost = cfg.tilt_progress_reward_per_rad*tilt+cfg.yaw_progress_reward_per_rad*yaw.abs()
    return -cost.clamp(max=1.)*captured

def terminal_flags(seated, force, moment, far, timeout, running, cfg):
    """Unfiltered measured load wins over a seat in the same 1 ms sample."""
    code = (force > cfg.force_abort_n).long() | ((moment > cfg.moment_abort_nm).long()*2) | (far.long()*4)
    code *= running.long()
    fault = (code != 0) & running
    success = seated & running & ~fault
    timed_out = timeout & running & ~fault & ~success
    return success, fault, timed_out, code


class CartesianTask:
    def __init__(self, *args, **kwargs):
        from .wrench_sensor import configure_v6_wrench_sensor
        configure_v6_wrench_sensor()
        super().__init__(*args, **kwargs)

    v6_buffers = ("wrench_tool", "filtered_wrench", "interval_wrench_sum", "interval_mean_wrench",
                  "interval_peak_force", "interval_peak_moment", "peak_moment", "max_legal_depth",
                  "alignment_phi", "interval_substeps", "physics_steps", "force_exceed_steps",
                  "moment_exceed_steps", "geometry_reached", "capture_time", "capture_depth",
                  "capture_xy", "capture_tilt", "capture_error", "capture_inside")

    def _make_reset_bank(self):
        super()._make_reset_bank()
        cfg = self.cfg
        self.control = CartesianTipImpedance(cfg, self.model, self.device, self.num_envs)
        self.torque_adr = int(self.model.sensor("robot/ft_frame_SENSOR_TORQUE").adr[0])
        self.payload_id = self.model.body("robot/hand_mount").id
        self.gravity_world = torch.tensor(self.model.opt.gravity.copy(), device=self.device, dtype=torch.float32)
        self.comp_qpos_ids, self.comp_dof_ids = passive_indices(self.model)
        self.comp_travel = torch.tensor([cfg.wrist_compliance_xy_travel_m]*2+[cfg.wrist_compliance_z_travel_m],
                                        device=self.device)
        # Nominal gravity lever calibrated at zero passive deflection; the
        # passive state is never used to build an actor input.
        cpu = mujoco.MjData(self.model)
        cpu.qpos[:] = self.q_nominal_np
        cpu.qpos[self.comp_qpos_ids] = 0
        mujoco.mj_forward(self.model, cpu)
        rotation = cpu.site_xmat[self.sensor_site].reshape(3, 3)
        com = rotation.T@(cpu.subtree_com[self.payload_id]-cpu.site_xpos[self.sensor_site])
        self.nominal_payload_com = torch.tensor(com, device=self.device, dtype=torch.float32)
        self.wrench_scale = torch.tensor([cfg.wrench_force_scale_n]*3+[cfg.wrench_moment_scale_nm]*3,
                                         device=self.device)
        n = self.num_envs
        for name in self.v6_buffers:
            shape = (n, 6) if "wrench" in name else (n,)
            dtype = torch.long if name in ("physics_steps", "force_exceed_steps", "moment_exceed_steps") else torch.float32
            if name in ("geometry_reached", "capture_inside"):
                dtype = torch.bool
            setattr(self, name, torch.zeros(shape, device=self.device, dtype=dtype))
        self.wrench_alpha = 1-math.exp(-cfg.physics_dt/cfg.governor_wrench_tau_s)

    # ----- measurement -------------------------------------------------
    def read_wrench(self):
        d = self.data
        raw = torch.cat((d.sensordata[:, self.force_adr:self.force_adr+3],
                         d.sensordata[:, self.torque_adr:self.torque_adr+3]), -1)
        sr = d.site_xmat[:, self.sensor_site].reshape(-1, 3, 3)
        tr = d.site_xmat[:, self.control.tip_id].reshape(-1, 3, 3)
        sp, tp = d.site_xpos[:, self.sensor_site], d.site_xpos[:, self.control.tip_id]
        compensated = compensate_payload_gravity(raw, sr, self.nominal_payload_com.expand(self.num_envs, -1),
                                                 self.payload_mass, self.gravity_world)
        offset = (sr.transpose(1, 2)@(tp-sp)[:, :, None]).squeeze(-1)
        world = transform_wrench_sensor_to_target_batch(compensated, offset, sr)
        tool = torch.cat(((tr.transpose(1, 2)@world[:, :3, None]).squeeze(-1),
                          (tr.transpose(1, 2)@world[:, 3:, None]).squeeze(-1)), -1)
        if not torch.isfinite(tool).all():
            raise FloatingPointError("non-finite wrench invalidates the run")
        return tool

    def proprioception(self):
        return torch.cat((super().proprioception(), self.interval_mean_wrench/self.wrench_scale), -1)

    def events(self, points, rotation, shaft_error):
        if self.shape == "circle":
            return circle_events(points, self.radius, rotation, shaft_error, self.cfg)
        return polygon_events(self.geometry, points, rotation, shaft_error, self.cfg)

    def tilt_yaw(self, rotation):
        tilt = torch.acos((-rotation[:, 2, 2]).clamp(-1., 1.))
        if self.shape == "circle":
            return tilt, torch.zeros_like(tilt)
        return tilt, self.geometry.yaw_error(rotation)

    def pose_target(self, center, rotation):
        _, yaw = self.tilt_yaw(rotation)
        mask = torch.full_like(yaw, 0. if self.shape == "circle" else 1.)
        half = math.pi if self.shape == "circle" else self.geometry.period/2
        return torch.cat((center/.01, rotation[:, :2, 2]/.05, (yaw/half)[:, None], mask[:, None]), -1)

    def observations(self):
        obs = super().observations()
        _, center, rotation = self.local_geometry()
        passive = self.data.qpos[:, self.comp_qpos_ids]/self.comp_travel
        p, r = self.control.pose(self.data)
        rt = r.transpose(1, 2)
        reference = torch.cat(((rt@(self.control.ref_pos-p)[:, :, None]).squeeze(-1)/self.cfg.cart_translation_lead_m,
                               (rt@so3_log(self.control.ref_rot@rt)[:, :, None]).squeeze(-1)/self.cfg.cart_rotation_lead_rad), -1)
        # Persistent governor/reward state belongs in the critic, never in the
        # history consumed by the blind actor.
        obs["critic"] = torch.cat((obs["critic"], (self.max_legal_depth/self.cfg.goal_depth_m)[:, None],
                                   passive, reference, self.filtered_wrench/self.wrench_scale,
                                   self.alignment_phi[:, None]), -1)
        obs["pose"] = self.pose_target(center, rotation)
        return obs

    # ----- episode management ------------------------------------------
    def reset(self, ids=None):
        if self.history.shape[-1] != FRAME_DIM:
            self.history = torch.zeros(self.num_envs, self.cfg.history_steps, FRAME_DIM, device=self.device)
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        if not len(ids):
            return
        super().reset(ids)
        self.control.reset(ids, self.data)
        tool = self.read_wrench()
        for name in ("wrench_tool", "filtered_wrench", "interval_mean_wrench"):
            getattr(self, name)[ids] = tool[ids]
        for name in self.v6_buffers:
            if name not in ("wrench_tool", "filtered_wrench", "interval_mean_wrench"):
                getattr(self, name)[ids] = 0
        self.capture_time[ids] = -1
        self.peak_force[ids] = 0
        self.peak_moment[ids] = 0
        self.history[ids] = self.proprioception()[ids, None, :].expand(-1, self.cfg.history_steps, -1)
        return self.observations()

    def monitor_physics(self, running, integration_wrench):
        StudyInsertionEnv.monitor_physics(self)
        tool = self.read_wrench()
        self.wrench_tool.copy_(tool)
        self.filtered_wrench.lerp_(tool, self.wrench_alpha)
        self.interval_wrench_sum += tool*running[:, None]
        self.interval_substeps += running
        self.interval_mean_wrench = self.interval_wrench_sum/self.interval_substeps.clamp_min(1)[:, None]
        # Retain the solver's pre-integration peak as well as the refreshed
        # end-of-substep measurement. One physical interval counts only once.
        force = torch.maximum(tool[:, :3].norm(dim=-1), integration_wrench[:, :3].norm(dim=-1))
        moment = torch.maximum(tool[:, 3:].norm(dim=-1), integration_wrench[:, 3:].norm(dim=-1))
        self.interval_peak_force = torch.maximum(self.interval_peak_force, force*running)
        self.interval_peak_moment = torch.maximum(self.interval_peak_moment, moment*running)
        self.peak_force = torch.maximum(self.peak_force, force*running)
        self.peak_moment = torch.maximum(self.peak_moment, moment*running)
        self.force_exceed_steps += (force > self.cfg.force_abort_n) & running
        self.moment_exceed_steps += (moment > self.cfg.moment_abort_nm) & running
        return force, moment

    def _sample(self):
        points, center, rotation = self.local_geometry()
        _, depth, error, _, _ = self.truth()
        tilt, yaw = self.tilt_yaw(rotation)
        captured, seated = self.events(points, rotation, error)
        return dict(depth=depth, shaft_error=error, xy_error=center[:, :2].norm(dim=-1),
                    tilt_rad=tilt, yaw_error_rad=yaw, capture_now=captured, seated=seated,
                    full_ring_inside=self.inside_bore(points).all(-1),
                    far=center.norm(dim=-1)>self.cfg.far_distance_m)

    def step(self, action):
        cfg = self.cfg
        if not torch.isfinite(action).all():
            raise FloatingPointError("non-finite action")
        active = self.active.clone()
        captured_before = self.captured.clone()
        old_depth, old_phi = self.max_legal_depth.clone(), self.alignment_phi.clone()
        self.control.set_action(self.data, torch.where(active[:, None], action, 0))
        done, success, fault, timeout = [torch.zeros_like(active) for _ in range(4)]
        cached_obs, cached_sample, cached_qpos = None, None, None
        for name in ("interval_wrench_sum", "interval_peak_force", "interval_peak_moment", "interval_substeps"):
            getattr(self, name).zero_()
        for _ in range(cfg.decimation):
            running = active & ~done
            self.control.govern(self.data, self.filtered_wrench, stopped=~running)
            self.control.apply(self.data)
            self.sim.step()
            integration_wrench = self.read_wrench()
            # mjwarp.step leaves derived poses/sensors at the pre-integration
            # state. Refresh before testing a 1 ms end state or reading F/T.
            self.sim.forward()
            force, moment = self.monitor_physics(running, integration_wrench)
            self.physics_steps += running.long()
            sample = self._sample()
            hit, bad, expired, code = terminal_flags(
                sample["seated"], force, moment, sample["far"],
                self.physics_steps >= math.ceil(cfg.horizon_s/cfg.physics_dt), running, cfg)
            self.fault_code |= code
            self.geometry_reached |= sample["seated"] & running
            first = sample["capture_now"] & ~self.captured & running & ~bad
            self.captured |= sample["capture_now"] & running & ~bad
            self.capture_time[first] = self.physics_steps[first]*cfg.physics_dt
            for target, key in (("capture_depth", "depth"), ("capture_xy", "xy_error"),
                                ("capture_tilt", "tilt_rad"), ("capture_error", "shaft_error"),
                                ("capture_inside", "full_ring_inside")):
                getattr(self, target)[first] = sample[key][first]
            legal_depth = torch.where(self.captured & running & (sample["shaft_error"] <= cfg.geometry_tolerance_m),
                                      sample["depth"].clamp(0., cfg.goal_depth_m), self.max_legal_depth)
            self.max_legal_depth = torch.maximum(self.max_legal_depth, legal_depth)
            next_phi = alignment_potential(sample["tilt_rad"], sample["yaw_error_rad"], self.captured, cfg)
            old_phi = torch.where(first, next_phi, old_phi)
            next_phi = torch.where(hit, 0., next_phi)
            self.alignment_phi = torch.where(running, next_phi, self.alignment_phi)
            new_done = hit | bad | expired
            if new_done.any():
                # Cache the first terminal physical state before cancelling its
                # reference. PPO bootstraps this state only for a time limit.
                current = self.observations()
                current["history"] = torch.cat((self.history[:, 1:], self.proprioception()[:, None]), 1)
                if cached_obs is None:
                    cached_obs = {k:v.clone() for k,v in current.items()}
                    cached_sample = {k:v.clone() for k,v in sample.items()}
                    cached_qpos = self.data.qpos.clone()
                else:
                    for key, value in current.items():
                        cached_obs[key][new_done] = value[new_done]
                    for key, value in sample.items():
                        cached_sample[key][new_done] = value[new_done]
                    cached_qpos[new_done] = self.data.qpos[new_done]
                self.control.cancel_reference(self.data, new_done)
            done |= new_done
            success |= hit
            fault |= bad
            timeout |= expired
        if not torch.isfinite(self.data.qpos).all() or not torch.isfinite(self.data.qvel).all():
            raise FloatingPointError("non-finite simulation")
        if self.buffer_overflow.any():
            raise RuntimeError("contact/constraint overflow invalidates run")
        self.steps += active.long()
        terminal = self.observations()
        terminal["history"] = torch.cat((self.history[:, 1:], self.proprioception()[:, None]), 1)
        qpos = self.data.qpos.clone()
        if cached_obs is not None:
            for key in terminal:
                terminal[key][done] = cached_obs[key][done]
            for key in sample:
                sample[key][done] = cached_sample[key][done]
            qpos[done] = cached_qpos[done]
        self.history = terminal["history"].clone()
        first = self.captured & ~captured_before & active
        progress = (self.max_legal_depth-old_depth).clamp_min(0.)
        axial = -self.interval_mean_wrench[:, 2]
        lo, hi = cfg.preload_force_range_n
        preload = (~self.captured) & (axial >= lo) & (axial <= hi) & ~fault
        overload = (cfg.overload_force_weight*(self.interval_peak_force-cfg.overload_force_n).clamp_min(0).square()
                    + cfg.overload_moment_weight*(self.interval_peak_moment-cfg.overload_moment_nm).clamp_min(0).square()
                    ).clamp(max=cfg.overload_penalty_cap)
        # Gamma-consistent potential, zero before capture and at absorbing
        # terminals; the finite time limit retains its bootstrap potential.
        shaping = cfg.gamma*self.alignment_phi-old_phi
        reward = (first.float()+success.float()+cfg.depth_progress_reward*progress/cfg.goal_depth_m
                  + shaping+cfg.preload_bonus*preload.float()-overload-cfg.load_fault_penalty*fault)*active
        self.episode_return += reward
        info = dict(success=success, fault=fault, timeout=timeout,
                    **{k:v for k,v in sample.items() if k not in ("seated", "far")},
                    new_milestones=first.long(), captured=self.captured.clone(),
                    geometry_success=self.geometry_reached.clone(), force_qualified_success=success.clone(),
                    force_peak_n=self.peak_force.clone(), moment_peak_nm=self.peak_moment.clone(),
                    force_exceed_s=self.force_exceed_steps*cfg.physics_dt,
                    moment_exceed_s=self.moment_exceed_steps*cfg.physics_dt,
                    elapsed_s=self.physics_steps*cfg.physics_dt, trajectory_qpos=qpos,
                    capture_time_s=self.capture_time.clone(), capture_depth_m=self.capture_depth.clone(),
                    capture_xy_m=self.capture_xy.clone(), capture_tilt_rad=self.capture_tilt.clone(),
                    capture_shaft_error_m=self.capture_error.clone(), capture_full_ring_inside=self.capture_inside.clone(),
                    fault_code=self.fault_code.clone(), return_=self.episode_return.clone(), terminal=terminal,
                    preload=preload & active, overload_cost=overload*active,
                    legal_depth_m=self.max_legal_depth.clone(),
                    yaw_alignment_required=torch.full_like(success, self.shape != "circle"))
        self.active &= ~done
        if self.auto_reset:
            self.reset(done.nonzero().flatten())
        return self.observations(), reward, done, info

    def snapshot(self):
        state = super().snapshot()
        state["v6"] = {name: getattr(self, name).clone() for name in self.v6_buffers}
        state["v6_control"] = dict(ref_pos=self.control.ref_pos.clone(), ref_rot=self.control.ref_rot.clone())
        return state

    def restore_snapshot(self, state):
        super().restore_snapshot(state)
        for name, value in state["v6"].items():
            getattr(self, name).copy_(value)
        self.control.ref_pos.copy_(state["v6_control"]["ref_pos"])
        self.control.ref_rot.copy_(state["v6_control"]["ref_rot"])


class CartesianCircleEnv(CartesianTask, CircleStudyEnv):
    pass


class CartesianPolygonEnv(CartesianTask, PolygonStudyEnv):
    pass


def make_cartesian_shape_environment(cfg, **kwargs):
    shape, = cfg.assembly_shapes
    cls = CartesianCircleEnv if shape == "circle" else CartesianPolygonEnv
    return cls(cfg, **kwargs)


class CartesianMixedEnv(MixedStudyEnv):
    """Same grouping as revision 4, with revision-6 shape groups."""

    def __init__(self, cfg, device="cuda:0", auto_reset=True):
        self.cfg = cfg.validate()
        self.device = device
        self.num_envs = cfg.num_envs
        self.shapes = cfg.assembly_shapes
        self.auto_reset = auto_reset
        size = cfg.num_envs//len(self.shapes)
        self.envs = [make_cartesian_shape_environment(
            replace(cfg, assembly_shapes=[shape], num_envs=size,
                    reset_bank_size=cfg.reset_bank_size//len(self.shapes), seed=cfg.seed+1000*i),
            device=device, auto_reset=auto_reset) for i, shape in enumerate(self.shapes)]
        self.shape_ids = torch.arange(len(self.shapes), device=device).repeat_interleave(size)
        vocabulary = cfg.shape_vocabulary
        code_ids = torch.tensor([vocabulary.index(s) for s in self.shapes], device=device).repeat_interleave(size)
        self.shape_code = torch.nn.functional.one_hot(code_ids, len(vocabulary)).float()
        self.critic_dim = FRAME_DIM+3+9+192+1+1+3+6+6+1+len(vocabulary)
        self.data = _ReadOnlyData(self)
        self.reset_audit = dict(size=cfg.reset_bank_size, all_starts_have_actual_contact=all(
            e.reset_audit["all_starts_have_actual_contact"] for e in self.envs),
            by_shape={s: e.reset_audit for s, e in zip(self.shapes, self.envs)},
            critic_shape_vocabulary=list(vocabulary), study_revision=6)

    def __getattr__(self, name):
        if name == "peak_moment":
            return torch.cat([e.peak_moment for e in self.envs])
        return super().__getattr__(name)


def make_task_environment(cfg, **kwargs):
    if cfg.study_revision == 6:
        return CartesianMixedEnv(cfg, **kwargs)
    from .study_environment import make_environment
    env = make_environment(cfg, **kwargs)
    if cfg.study_revision == 4 and cfg.seating_protocol == "chamfer_v1":
        from .seating_ablation import apply_seating_ablation
        return apply_seating_ablation(env)
    return env
