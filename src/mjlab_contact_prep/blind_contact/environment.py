"""Vectorized MuJoCo-Warp insertion; the actor receives only proprioception."""
import math
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
from mjlab_contact_prep.backend import configure_backend
from .assets import assembly_spec, peg_geometry, solve_ik, NOMINAL_TIP, NOMINAL_ROTATION
from .control import JointImpedance


def rotation_coordinates(matrix):
    """Small XYZ Euler angles relative to a fixed calibration frame."""
    return torch.stack((torch.atan2(matrix[:,2,1], matrix[:,2,2]),
                        torch.asin(-matrix[:,2,0].clamp(-1,1)),
                        torch.atan2(matrix[:,1,0], matrix[:,0,0])), -1)


def contact_geometry(position, rotation, hole, radius, clearance, count=64, band=.001):
    """Training/evaluation truth, isolated from the proprioceptive observation."""
    device = position.device
    angles = torch.arange(count, device=device)*2*math.pi/count
    ring = torch.stack((radius*angles.cos(), radius*angles.sin(), torch.zeros_like(angles)), -1)
    points = position[:,None,:] + ring[None] @ rotation.transpose(1,2) - hole[:,None,:]
    radial = points[:,:,:2].norm(dim=-1)
    z = points[:,:,2]
    # Conservative circular aperture, within the actual circumscribed bore.
    allowed = radius+clearance+z.clamp(-.001,0)+.001
    labels = torch.zeros_like(z, dtype=torch.long)
    near = z < band
    labels[near] = 1
    labels[near & (radial <= allowed)] = 2
    depth = -z.max(-1).values
    center = position-hole
    axis = rotation[:,:,2]
    cosine = (-axis[:,2]).clamp_min(.1)
    # Inspect every change in bore slope; axis drift between levels is linear.
    heights = torch.tensor([0.,-.001,-.02,-.0395,-.04],device=device)
    t = (heights[None]-center[:,2,None])/axis[:,2,None].clamp(max=-.1)
    centers_xy = center[:,None,:2] + t[:,:,None]*axis[:,None,:2]
    permitted = torch.tensor([radius+clearance+.001, radius+clearance,
                              radius+clearance, radius+clearance,
                              radius+clearance+.0005],device=device)
    # t is negative along the held shaft (local -Z).
    engaged = (t <= 0) & (t >= -.05)
    excess = centers_xy.norm(dim=-1) + radius/cosine[:,None] - permitted
    excess = torch.where(engaged, excess, -torch.ones_like(excess))
    shaft_error = excess.max(-1).values
    return labels, depth, shaft_error, center, axis


class BlindInsertionEnv:
    observation_dim = 30
    critic_dim = 30 + 6 + 64*3 + 1

    def __init__(self, cfg, device="cuda:0", *, auto_reset=True):
        self.cfg = cfg.validate()
        self.device = device
        self.num_envs = cfg.num_envs
        self.auto_reset = auto_reset
        configure_backend(cfg.physics_backend)
        self.spec = assembly_spec(cfg)
        self.model = self.spec.compile()
        self.sim = Simulation(cfg.num_envs,
            SimulationCfg(nconmax=cfg.nconmax, njmax=cfg.njmax,
                mujoco=MujocoCfg(timestep=cfg.physics_dt, integrator="implicitfast",
                                solver="newton", iterations=50, cone="pyramidal",
                                ccd_iterations=50)),
            self.model, device)
        self.data = self.sim.data
        self.control = JointImpedance(cfg, self.model, device, cfg.num_envs)
        self.tip_id = self.model.site("robot/peg_geometry_tip_site").id
        self.hole_id = self.model.site("hole/hole_mouth_site").id
        self.hole_mocap = int(self.model.body("hole/root_hole").mocapid[0])
        self.lin_adr = int(self.model.sensor("robot/tcp_linear_velocity").adr[0])
        self.ang_adr = int(self.model.sensor("robot/tcp_angular_velocity").adr[0])
        self.force_adr = int(self.model.sensor("robot/ft_frame_SENSOR_FORCE").adr[0])
        self.sensor_site = self.model.site("robot/ft_frame_SITE").id
        self.payload_mass = float(self.model.body_subtreemass[self.model.body("robot/hand_mount").id])
        self.radius = peg_geometry()["radius_m"]
        self.q_nominal_np = solve_ik(self.model, NOMINAL_TIP)
        self.q_nominal = torch.tensor(self.q_nominal_np[self.control.qids],dtype=torch.float32,device=device)
        self.p_nominal = torch.tensor(NOMINAL_TIP, dtype=torch.float32, device=device)
        self.r_nominal = torch.tensor(NOMINAL_ROTATION, dtype=torch.float32, device=device)
        self.rng = torch.Generator(device=device).manual_seed(cfg.seed)
        self._make_reset_bank()
        self.history = torch.zeros(cfg.num_envs, cfg.history_steps, 30, device=device)
        self.steps = torch.zeros(cfg.num_envs,device=device,dtype=torch.long)
        self.dwell = torch.zeros(cfg.num_envs,device=device,dtype=torch.long)
        self.active = torch.ones(cfg.num_envs,device=device,dtype=torch.bool)
        self.milestones = torch.tensor(cfg.milestone_depths_m, device=device)
        self.bonuses = torch.tensor(cfg.milestone_rewards, device=device)
        self.reached = torch.zeros(cfg.num_envs,len(cfg.milestone_depths_m),device=device,dtype=torch.bool)
        self.episode_return = torch.zeros(cfg.num_envs,device=device)
        self.peak_constraint = torch.zeros(cfg.num_envs,device=device)
        self.peak_force = torch.zeros(cfg.num_envs,device=device)
        self.fault_code = torch.zeros(cfg.num_envs,device=device,dtype=torch.long)
        self.buffer_overflow = torch.zeros(1,device=device,dtype=torch.bool)
        self.max_contacts = torch.zeros(1,device=device,dtype=torch.long)
        self.max_constraints = torch.zeros(1,device=device,dtype=torch.long)
        self.bore_levels = torch.tensor([0.,-.001,-.02,-.0395,-.04],device=device)
        self.bore_radii = self.radius+cfg.clearance_per_side_m+torch.tensor([.001,0.,0.,0.,.0005],device=device)
        self.reset_radius = torch.zeros(cfg.num_envs,device=device)
        self.reset()

    def _make_reset_bank(self):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        bank = []
        for _ in range(cfg.reset_bank_size):
            p = NOMINAL_TIP.copy()
            p[:2] += rng.uniform(-cfg.world_xy_jitter_m, cfg.world_xy_jitter_m, 2)
            p[2] = .1+cfg.initial_gap_m
            tilt = rng.uniform(-cfg.tilt_deg, cfg.tilt_deg, 2)
            r = Rotation.from_euler("xy", tilt, degrees=True).as_matrix() @ NOMINAL_ROTATION
            bank.append(solve_ik(self.model, p, r, self.q_nominal_np[:6]))
        self.reset_bank = torch.tensor(np.array(bank), device=self.device, dtype=torch.float32)

    def proprioception(self):
        q = self.data.qpos[:,self.control.qids]
        dq = self.data.qvel[:,self.control.vids]
        p = self.data.site_xpos[:,self.tip_id]
        r = self.data.site_xmat[:,self.tip_id].reshape(-1,3,3)
        orientation = rotation_coordinates(self.r_nominal.T[None] @ r)
        linear = self.data.sensordata[:,self.lin_adr:self.lin_adr+3]
        angular = self.data.sensordata[:,self.ang_adr:self.ang_adr+3]
        return torch.cat(((q-self.q_nominal)/.1, dq/.5,
                          (p-self.p_nominal)/.05, orientation/.1,
                          linear/.1, angular/.5, self.control.last_action),-1)

    def truth(self):
        return contact_geometry(self.data.site_xpos[:,self.tip_id],
                                self.data.site_xmat[:,self.tip_id].reshape(-1,3,3),
                                self.data.site_xpos[:,self.hole_id],
                                self.radius, self.cfg.clearance_per_side_m,
                                self.cfg.contact_points, self.cfg.contact_band_m)

    def observations(self):
        labels, depth, shaft_error, center, axis = self.truth()
        proprio = self.proprioception()
        critic = torch.cat((proprio, center/.05, axis,
                            torch.nn.functional.one_hot(labels,3).flatten(1).float(),
                            (self.steps*self.cfg.policy_dt/self.cfg.horizon_s)[:,None]),-1)
        return dict(history=self.history.clone(), critic=critic, labels=labels)

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        self.sim.reset(ids)
        choice = torch.randint(len(self.reset_bank),(len(ids),),device=self.device,generator=self.rng)
        self.data.qpos[ids] = self.reset_bank[choice]
        self.data.qvel[ids] = 0
        self.control.last_action[ids] = 0
        self.control.target[ids] = self.data.qpos[ids][:,self.control.qids]
        self.data.mocap_quat[ids,self.hole_mocap] = torch.tensor([1.,0,0,0],device=self.device)
        self.sim.forward()
        p = self.data.site_xpos[ids,self.tip_id].clone()
        u = torch.rand(len(ids),device=self.device,generator=self.rng)
        radial = (self.cfg.xy_min_m**2 + u*(self.cfg.xy_max_m**2-self.cfg.xy_min_m**2)).sqrt()
        angle = torch.rand(len(ids),device=self.device,generator=self.rng)*2*math.pi
        p[:,0] += radial*angle.cos()
        p[:,1] += radial*angle.sin()
        p[:,2] = .1
        self.data.mocap_pos[ids,self.hole_mocap] = p
        self.sim.forward()
        self.steps[ids] = 0
        self.dwell[ids] = 0
        self.reached[ids] = False
        self.active[ids] = True
        self.episode_return[ids] = 0
        self.peak_constraint[ids] = 0
        self.peak_force[ids] = 0
        self.fault_code[ids] = 0
        self.reset_radius[ids] = radial
        _,depth,error,_,_=self.truth()
        if (depth[ids]>=self.cfg.goal_depth_m).any():
            raise RuntimeError("reset is already successful; reject this configuration")
        if self.cfg.initial_gap_m<0 and (error[ids]>self.cfg.geometry_tolerance_m).any():
            raise RuntimeError("invalid penetrated reset")
        # Captured reset curricula cannot collect an already passed milestone.
        self.reached[ids] = (depth[ids,None]>=self.milestones)&(error[ids,None]<=self.cfg.geometry_tolerance_m)
        # Repeat the fresh measured observation; never retain another episode.
        self.history[ids] = self.proprioception()[ids,None,:].expand(-1,self.cfg.history_steps,-1)
        return self.observations()

    def physical_status(self):
        """Conservative whole-shaft containment and protection at each 1 ms sample."""
        center = self.data.site_xpos[:,self.tip_id]-self.data.site_xpos[:,self.hole_id]
        axis = self.data.site_xmat[:,self.tip_id].reshape(-1,3,3)[:,:,2]
        depth = -center[:,2]-self.radius*axis[:,:2].norm(dim=-1)
        cosine = (-axis[:,2]).clamp_min(.1)
        t = (self.bore_levels[None]-center[:,2,None])/axis[:,2,None].clamp(max=-.1)
        centers = center[:,None,:2]+t[:,:,None]*axis[:,None,:2]
        excess = centers.norm(dim=-1)+self.radius/cosine[:,None]-self.bore_radii
        shaft_error = torch.where((t<=0)&(t>=-.05),excess,-torch.ones_like(excess)).max(-1).values
        speed = self.data.sensordata[:,self.lin_adr:self.lin_adr+3].norm(dim=-1)
        sensor_rot = self.data.site_xmat[:,self.sensor_site].reshape(-1,3,3)
        gravity_sensor = -9.81*self.payload_mass*sensor_rot[:,2,:]
        force = (-self.data.sensordata[:,self.force_adr:self.force_adr+3]-gravity_sensor).norm(dim=-1)
        velocity_fault = self.data.qvel[:,self.control.vids].abs().max(-1).values > 2.
        pose_fault = ((center[:,:2].norm(dim=-1)>.04)|(center[:,2]>.05)
                      |(depth>.048)|(axis[:,2]>-.95))
        code = ((force>self.cfg.force_abort_n).long()+2*velocity_fault.long()+4*pose_fault.long())
        qualified = ((depth>=self.cfg.goal_depth_m)&(shaft_error<=self.cfg.geometry_tolerance_m)
                     &(speed<=self.cfg.success_speed_mps)&(code==0))
        return qualified, code, force, depth, shaft_error

    def monitor_physics(self):
        qualified, code, force, _, _ = self.physical_status()
        self.fault_code |= code*self.active.long()
        self.peak_force = torch.maximum(self.peak_force,force*self.active)
        self.dwell = torch.where(qualified&(self.fault_code==0)&self.active,self.dwell+1,0)
        self.max_contacts = torch.maximum(self.max_contacts,self.data.nacon.long())
        self.max_constraints = torch.maximum(self.max_constraints,self.data.nefc.max().reshape(1).long())
        self.buffer_overflow |= ((self.data.nacon>=self.sim.wp_data.naconmax)
                                 |(self.data.nefc.max()>=self.sim.wp_data.njmax))

    def step(self, action):
        if not torch.isfinite(action).all():
            raise FloatingPointError("non-finite policy action")
        cfg = self.cfg
        action = torch.where(self.active[:,None],action,0)
        self.control.set_action(self.data.qpos[:,self.control.qids],action)
        for _ in range(cfg.decimation):
            # Samples are at integration boundaries. Latch any short violation;
            # an apparently stable 50 Hz endpoint cannot erase it.
            self.monitor_physics()
            self.control.target[:] = torch.where((self.fault_code!=0)[:,None],
                self.data.qpos[:,self.control.qids],self.control.target)
            self.control.apply(self.data)
            self.sim.step()
        self.sim.forward()  # Refresh FK/sensors after the last integration.
        if not torch.isfinite(self.data.qpos).all() or not torch.isfinite(self.data.qvel).all():
            raise FloatingPointError("non-finite physical state")
        if self.buffer_overflow.any():
            raise RuntimeError("MuJoCo-Warp contact/constraint buffer overflow invalidates this run")
        self.steps += self.active.long()
        labels, depth, shaft_error, center, axis = self.truth()
        # Joint reaction is used only for diagnostics, never as actor input.
        reaction = self.data.qfrc_constraint[:,self.control.vids].norm(dim=-1)
        self.peak_constraint = torch.maximum(self.peak_constraint, reaction)
        qualified, code, force, depth, shaft_error = self.physical_status()
        self.fault_code |= code*self.active.long()
        self.peak_force = torch.maximum(self.peak_force,force*self.active)
        fault = self.fault_code!=0
        self.dwell = torch.where(qualified&~fault,self.dwell,0)
        success = (self.dwell >= math.ceil(cfg.success_hold_s/cfg.physics_dt)) & ~fault
        timeout = self.steps >= math.ceil(cfg.horizon_s/cfg.policy_dt)
        done = (success | fault | timeout) & self.active
        crossed = ((depth[:,None] >= self.milestones) & ~self.reached
                   & (shaft_error[:,None] <= cfg.geometry_tolerance_m)
                   & (self.milestones <= cfg.goal_depth_m) & ~fault[:,None])
        reward = (crossed*self.bonuses).sum(-1) + 10*success.float() - 2*fault.float() -.001
        reward *= self.active
        self.reached |= crossed
        self.episode_return += reward
        self.history = torch.cat((self.history[:,1:],self.proprioception()[:,None,:]),1)
        terminal_obs = self.observations()
        info = dict(success=success.clone(), fault=fault.clone(), timeout=(timeout & ~success & ~fault),
                    depth=depth.clone(), shaft_error=shaft_error.clone(),
                    new_milestones=(crossed & self.active[:,None]).sum(-1),
                    force_peak_n=self.peak_force.clone(),fault_code=self.fault_code.clone(),
                    return_=self.episode_return.clone(), terminal=terminal_obs)
        self.active &= ~done
        if self.auto_reset and done.any():
            self.reset(done.nonzero().flatten())
        return self.observations(), reward, done, info

    def snapshot(self):
        data = {key:getattr(self.data,key).clone() for key in
                ["qpos","qvel","act","ctrl","qacc_warmstart","mocap_pos","mocap_quat","time"]}
        buffers = {key:getattr(self,key).clone() for key in
                   ["history","steps","dwell","active","reached","episode_return",
                    "peak_constraint","peak_force","fault_code","buffer_overflow",
                    "max_contacts","max_constraints","reset_radius"]}
        return dict(data=data,buffers=buffers,target=self.control.target.clone(),
                    last_action=self.control.last_action.clone(),rng=self.rng.get_state(),
                    reset_bank=self.reset_bank.clone())

    def restore_snapshot(self, state):
        self.sim.reset()
        for key,value in state["data"].items():
            getattr(self.data,key).copy_(value)
        for key,value in state["buffers"].items():
            getattr(self,key).copy_(value)
        self.control.target.copy_(state["target"])
        self.control.last_action.copy_(state["last_action"])
        self.rng.set_state(state["rng"].cpu())
        self.reset_bank.copy_(state["reset_bank"])
        self.sim.forward()
        self.data.qacc_warmstart.copy_(state["data"]["qacc_warmstart"])

    def close(self):
        pass
