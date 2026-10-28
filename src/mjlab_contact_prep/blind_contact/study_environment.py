"""Study recipe on the user's fixed, rigidly mounted circular assembly.

The public demo supplies point-label and seating semantics. The 40 mm seat,
contact-reset bank, local fixture workspace and first-capture threshold are
explicit adaptations to the Estun asset, not claimed original training code.
"""
import math
import time
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation
from .assets import NOMINAL_TIP, NOMINAL_ROTATION, solve_ik
from .environment import BlindInsertionEnv, contact_geometry, rotation_coordinates


def edge_labels(signed_distance, height, band=.001):
    near_surface = height.abs() <= band
    near_edge = signed_distance.abs() <= band
    labels = torch.zeros_like(height, dtype=torch.long)
    labels[near_surface & (signed_distance > band)] = 1
    inside = ((height <= band) & (signed_distance < -band)) | ((height <= 0) & near_edge)
    labels[inside] = 2
    labels[near_surface & near_edge] = 1
    return labels


def study_events(points, radius, cfg, shaft_error=None):
    radial = points[..., :2].norm(dim=-1)
    height = points[..., 2]
    inside_bore = radial <= radius + cfg.clearance_per_side_m
    captured = (inside_bore & (height < -cfg.capture_depth_m)).any(-1)
    seated = (inside_bore & ((height + cfg.goal_depth_m).abs() <= cfg.seating_tolerance_m)).all(-1)
    if cfg.study_revision >= 2:
        if shaft_error is None:
            raise ValueError("study revision 2 requires whole-shaft geometry")
        # The ordered ring's normal is the peg's local +Z. The shaft helper
        # assumes it points into the hole; explicitly reject an inverted peg,
        # even when no bore sample engages and its error uses the -1 sentinel.
        u, v = points[:,1,:2]-points[:,0,:2], points[:,2,:2]-points[:,0,:2]
        downward = u[:,0]*v[:,1]-u[:,1]*v[:,0] < 0
        legal = (shaft_error <= cfg.geometry_tolerance_m) & downward
        # A useful capture has the complete lower rim past the entry chamfer,
        # with the engaged shaft contained in the bore. One dipped edge is not
        # an insertion stage, and cannot collect this once-per-episode reward.
        captured = inside_bore.all(-1) & (height.max(-1).values <= -cfg.capture_depth_m) & legal
        seated &= legal
    return captured, seated


def event_reward(captured, already_captured, seated, done_before, far):
    first = captured & ~already_captured & ~done_before & ~far
    success = seated & ~done_before & ~far
    return first.float() + success.float(), first, success


class StudyInsertionEnv(BlindInsertionEnv):
    # Actual fields: proprio 30, relative position 3, rotation matrix 9,
    # one-hot labels 192. No padding to imitate another robot's dimensions.
    critic_dim = 234
    nominal_rotation = NOMINAL_ROTATION

    def reset_outline(self):
        angles = np.arange(256)*2*np.pi/256
        return np.c_[self.radius*np.cos(angles), self.radius*np.sin(angles), np.zeros(256)]

    def reset_outside(self, xy):
        return np.linalg.norm(xy, axis=1) > self.radius+self.cfg.clearance_per_side_m+.001

    def edge_points(self):
        a = torch.arange(64,device=self.device)*2*math.pi/64
        return torch.stack((self.radius*a.cos(),self.radius*a.sin(),torch.zeros_like(a)),-1)

    def inside_bore(self, points):
        return points[...,:2].norm(dim=-1) <= self.radius+self.cfg.clearance_per_side_m

    def events(self, points, rotation, shaft_error):
        return study_events(points, self.radius, self.cfg, shaft_error)

    def _make_reset_bank(self):
        cfg = self.cfg
        self.critic_dim = 235 if cfg.study_revision >= 2 else 234
        self.tcp_id = self.model.site("robot/blind_tcp_site").id
        self.velocity_scale = torch.tensor(cfg.velocity_scale, device=self.device)
        self.hole_geoms_np = np.array([g for g in range(self.model.ngeom)
                                      if self.model.geom(g).name.startswith("hole/")])
        self.hole_geoms = torch.tensor(self.hole_geoms_np, device=self.device)
        self.sim.expand_model_fields(("geom_friction", "geom_solref"))
        self.captured = torch.zeros(cfg.num_envs, device=self.device, dtype=torch.bool)
        self.reset_choice = torch.zeros(cfg.num_envs, device=self.device, dtype=torch.long)
        rng = np.random.default_rng(cfg.seed)
        ring = self.reset_outline()
        d = mujoco.MjData(self.model)
        bank, holes, quats, radii, gaps = [], [], [], [], []
        attempts = 0
        limits = self.control.joint_ranges.cpu().numpy()
        started = time.monotonic()
        while len(bank) < cfg.reset_bank_size:
            attempts += 1
            if attempts > cfg.reset_bank_size*100:
                raise RuntimeError("could not construct enough valid contact starts")
            hole = np.array([*NOMINAL_TIP[:2], .1])
            hole[:2] += rng.uniform(-cfg.fixture_xy_jitter_m, cfg.fixture_xy_jitter_m, 2)
            hole[2] += rng.uniform(-cfg.fixture_z_jitter_m, cfg.fixture_z_jitter_m)
            rf = Rotation.from_euler("xyz", rng.uniform(-cfg.fixture_tilt_deg,
                                                       cfg.fixture_tilt_deg, 3), degrees=True)
            relative_rotation = Rotation.from_euler("xyz", rng.uniform(-cfg.tilt_deg,
                                                                      cfg.tilt_deg, 3), degrees=True).as_matrix() @ self.nominal_rotation
            xy = rng.uniform(-cfg.reset_xy_halfwidth_m, cfg.reset_xy_halfwidth_m, 2)
            bottom = ring @ relative_rotation.T
            outside = self.reset_outside(bottom[:, :2] + xy)
            if not outside.any():
                continue
            # A 10 µm preload permits an actual, bounded contact at reset.
            z = -bottom[outside, 2].min() - .00001
            position = hole + rf.apply(np.r_[xy, z])
            try:
                q = solve_ik(self.model, position, rf.as_matrix() @ relative_rotation, self.q_nominal_np[:6])
            except RuntimeError:
                continue
            if np.any(q[self.control.qids] < limits[:,0]) or np.any(q[self.control.qids] > limits[:,1]):
                continue
            quat = rf.as_quat()[[3,0,1,2]]
            d.qpos[:] = q
            d.mocap_pos[self.hole_mocap] = hole
            d.mocap_quat[self.hole_mocap] = quat
            mujoco.mj_forward(self.model, d)
            distances = [c.dist for c in d.contact[:d.ncon]
                         if c.geom1 in self.hole_geoms_np or c.geom2 in self.hole_geoms_np]
            if not distances or not -.000025 <= min(distances) <= 0:
                continue
            bank.append(q); holes.append(hole); quats.append(quat)
            radii.append(float(np.linalg.norm(xy))); gaps.append(float(min(distances)))
            if len(bank) % 1024 == 0:
                print(f"reset bank {len(bank)}/{cfg.reset_bank_size}", flush=True)
        tensor = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self.reset_bank = tensor(bank)
        self.hole_bank = tensor(holes)
        self.quat_bank = tensor(quats)
        self.radius_bank = tensor(radii)
        self.reset_audit = dict(size=len(bank), attempts=attempts,
                                min_contact_distance_m=min(gaps), max_contact_distance_m=max(gaps),
                                max_initial_radius_m=max(radii), seed=cfg.seed,
                                generation_s=time.monotonic()-started,
                                all_starts_have_actual_contact=True)
        self.bottom_ring = self.edge_points()

    def local_geometry(self):
        rh = self.data.site_xmat[:,self.hole_id].reshape(-1,3,3)
        rp = self.data.site_xmat[:,self.tip_id].reshape(-1,3,3)
        delta = self.data.site_xpos[:,self.tip_id]-self.data.site_xpos[:,self.hole_id]
        center = (rh.transpose(1,2) @ delta[:,:,None]).squeeze(-1)
        rotation = rh.transpose(1,2) @ rp
        points = center[:,None,:] + self.bottom_ring[None] @ rotation.transpose(1,2)
        return points, center, rotation

    def truth(self):
        points, center, rotation = self.local_geometry()
        labels = edge_labels(points[:,:,:2].norm(dim=-1)-self.radius-self.cfg.clearance_per_side_m-.001,
                             points[:,:,2], self.cfg.contact_band_m)
        _, depth, error, _, axis = contact_geometry(center, rotation, torch.zeros_like(center),
                                                    self.radius, self.cfg.clearance_per_side_m)
        return labels, depth, error, center, axis

    def proprioception(self):
        q = self.data.qpos[:,self.control.qids]
        dq = self.data.qvel[:,self.control.vids]
        ranges = self.control.joint_ranges.float()
        p = self.data.site_xpos[:,self.tcp_id]
        r = self.data.site_xmat[:,self.tcp_id].reshape(-1,3,3)
        linear = self.data.sensordata[:,self.lin_adr:self.lin_adr+3]
        angular = self.data.sensordata[:,self.ang_adr:self.ang_adr+3]
        local_angular = (r.transpose(1,2) @ angular[:,:,None]).squeeze(-1)
        return torch.cat(((q-ranges.mean(-1))/(ranges[:,1]-ranges[:,0])*2.,
                          dq/self.velocity_scale, p, rotation_coordinates(r),
                          linear, local_angular, self.control.last_action),-1)

    def observations(self):
        labels, _, _, center, _ = self.truth()
        _, _, rotation = self.local_geometry()
        critic = torch.cat((self.proprioception(), center, rotation.flatten(1),
                            torch.nn.functional.one_hot(labels,3).flatten(1).float()),-1)
        if self.cfg.study_revision >= 2:
            # Reward-state memory is privileged. The actor still only receives
            # encoder-derived proprioception and executed previous actions.
            critic = torch.cat((critic, self.captured[:,None].float()), -1)
        return dict(history=self.history.clone(), critic=critic, labels=labels)

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs,device=self.device)
        if not len(ids):
            return
        self.sim.reset(ids)
        choice = torch.randint(len(self.reset_bank),(len(ids),),device=self.device,generator=self.rng)
        self.reset_choice[ids] = choice
        self.data.qpos[ids] = self.reset_bank[choice]
        self.data.qvel[ids] = 0
        self.data.mocap_pos[ids,self.hole_mocap] = self.hole_bank[choice]
        self.data.mocap_quat[ids,self.hole_mocap] = self.quat_bank[choice]
        self.control.last_action[ids] = 0
        self.control.target[ids] = self.data.qpos[ids][:,self.control.qids]
        for field, bounds in (("geom_friction", self.cfg.plate_friction_range),
                              ("geom_solref", self.cfg.plate_timeconst_range_s)):
            u = torch.rand(len(ids),device=self.device,generator=self.rng)
            values = (math.log(bounds[0])+u*math.log(bounds[1]/bounds[0])).exp()
            getattr(self.sim.model,field)[ids[:,None],self.hole_geoms[None],0] = values[:,None]
        self.sim.forward()
        for name in ("steps","dwell","episode_return","peak_constraint","peak_force","fault_code"):
            getattr(self,name)[ids] = 0
        self.active[ids] = True
        self.reached[ids] = False
        points, _, rotation = self.local_geometry()
        initially_captured, _ = self.events(points, rotation, self.truth()[2])
        self.captured[ids] = initially_captured[ids]
        self.reset_radius[ids] = self.radius_bank[choice]
        self.history[ids] = self.proprioception()[ids,None,:].expand(-1,self.cfg.history_steps,-1)
        return self.observations()

    def monitor_physics(self):
        # Retain overflow detection at every physics step, without force shaping.
        self.max_contacts = torch.maximum(self.max_contacts,self.data.nacon.long())
        self.max_constraints = torch.maximum(self.max_constraints,self.data.nefc.max().reshape(1).long())
        self.buffer_overflow |= ((self.data.nacon>=self.sim.wp_data.naconmax)
                                 |(self.data.nefc.max()>=self.sim.wp_data.njmax))

    def step(self, action):
        if not torch.isfinite(action).all():
            raise FloatingPointError("non-finite action")
        self.control.set_action(self.data.qpos[:,self.control.qids],
                                torch.where(self.active[:,None],action,0))
        for _ in range(self.cfg.decimation):
            self.control.apply(self.data)
            self.sim.step()
            self.monitor_physics()
        self.sim.forward()
        if not torch.isfinite(self.data.qpos).all() or not torch.isfinite(self.data.qvel).all():
            raise FloatingPointError("non-finite simulation")
        if self.buffer_overflow.any():
            raise RuntimeError("contact/constraint overflow invalidates run")
        self.steps += self.active.long()
        points, center, rotation = self.local_geometry()
        _, depth, error, _, _ = self.truth()
        captured, seated = self.events(points,rotation,error)
        far = center.norm(dim=-1)>self.cfg.far_distance_m
        reward, first, success = event_reward(captured,self.captured,seated,~self.active,far)
        self.captured |= captured
        self.fault_code = torch.where(far & self.active,4,self.fault_code)
        timeout = self.steps >= math.ceil(self.cfg.horizon_s/self.cfg.policy_dt)
        done = (success|far|timeout)&self.active
        self.episode_return += reward
        # Diagnostic force only: never reward, terminate, or alter control.
        sensor_rot = self.data.site_xmat[:,self.sensor_site].reshape(-1,3,3)
        gravity = -9.81*self.payload_mass*sensor_rot[:,2,:]
        force = (-self.data.sensordata[:,self.force_adr:self.force_adr+3]-gravity).norm(dim=-1)
        self.peak_force = torch.maximum(self.peak_force,force*self.active)
        self.history = torch.cat((self.history[:,1:],self.proprioception()[:,None,:]),1)
        terminal = self.observations()
        info = dict(success=success, fault=far, timeout=timeout&~far&~success,
                    depth=depth, shaft_error=error, new_milestones=first.long(),
                    captured=self.captured.clone(), capture_now=captured,
                    xy_error=center[:,:2].norm(dim=-1),
                    tilt_rad=torch.acos((-rotation[:,2,2]).clamp(-1.,1.)),
                    full_ring_inside=self.inside_bore(points).all(-1),
                    force_peak_n=self.peak_force.clone(),fault_code=self.fault_code.clone(),
                    return_=self.episode_return.clone(),terminal=terminal)
        self.active &= ~done
        if self.auto_reset:
            self.reset(done.nonzero().flatten())
        return self.observations(),reward,done,info

    def snapshot(self):
        state = super().snapshot()
        state["study"] = {name:getattr(self,name).clone() for name in
                          ("captured","reset_choice","hole_bank","quat_bank","radius_bank")}
        state["model"] = {name:getattr(self.sim.model,name).clone()
                          for name in ("geom_friction","geom_solref")}
        state["reset_audit"] = self.reset_audit
        return state

    def restore_snapshot(self,state):
        for name,value in state["study"].items():
            setattr(self,name,value.clone())
        for name,value in state["model"].items():
            getattr(self.sim.model,name).copy_(value)
        self.reset_audit = state["reset_audit"]
        super().restore_snapshot(state)


def make_environment(cfg, **kwargs):
    if cfg.study_revision == 4:
        from .polygon_environment import MixedStudyEnv
        return MixedStudyEnv(cfg, **kwargs)
    if cfg.study_revision == 3:
        from .polygon_environment import PolygonStudyEnv, MixedStudyEnv
        cls = MixedStudyEnv if len(cfg.assembly_shapes)>1 else PolygonStudyEnv
        return cls(cfg, **kwargs)
    cls = StudyInsertionEnv if cfg.method == "study" else BlindInsertionEnv
    return cls(cfg, **kwargs)
