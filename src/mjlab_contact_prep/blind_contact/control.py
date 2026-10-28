"""Measured-position joint impedance targets, held for one policy interval."""
import numpy as np
import torch


class JointImpedance:
    def __init__(self, cfg, model, device, num_envs):
        self.cfg = cfg
        self.qids = [int(model.joint(f"robot/joint_{i}").qposadr[0]) for i in range(1,7)]
        self.vids = [int(model.joint(f"robot/joint_{i}").dofadr[0]) for i in range(1,7)]
        self.aids = [model.actuator(f"robot/joint_{i}").id for i in range(1,7)]
        self.kp = torch.tensor(cfg.kp, device=device)
        self.kd = torch.tensor(cfg.kd, device=device)
        self.limits = torch.tensor(cfg.torque_limits, device=device)
        self.friction = torch.tensor(model.dof_frictionloss[self.vids],device=device,dtype=torch.float32)
        self.joint_ranges = torch.tensor(np.array([model.joint(f"robot/joint_{i}").range
                                          for i in range(1,7)]), device=device)
        self.target = torch.zeros(num_envs, 6, device=device)
        self.last_action = torch.zeros_like(self.target)
        self.torque = torch.zeros_like(self.target)

    def set_action(self, q, action):
        # PPO stores/log-scores the raw Gaussian sample. This deterministic map
        # bounds the executed joint increment without clipping the likelihood.
        mapped = action.tanh() if self.cfg.action_transform == "tanh" else action
        executed = mapped if self.cfg.action_clip is None else mapped.clamp(
            -self.cfg.action_clip, self.cfg.action_clip)
        self.last_action.copy_(executed)
        self.target.copy_((q + self.cfg.action_scale*self.last_action)
                          .clamp(self.joint_ranges[:,0], self.joint_ranges[:,1]))

    def apply(self, data):
        q, dq = data.qpos[:, self.qids], data.qvel[:, self.vids]
        # Only compensate the residual; six robot links already have gravcomp=1.
        residual = (data.qfrc_bias-data.qfrc_gravcomp)[:, self.vids]
        error = self.target-q
        # Model-based dry-friction compensation uses encoder tracking error,
        # never a contact/force signal. Physical joint friction stays unchanged.
        friction = (self.friction*torch.tanh(error/self.cfg.friction_deadband_rad)
                    if self.cfg.friction_compensation else 0.)
        self.torque.copy_((self.kp*error-self.kd*dq+residual+friction)
                          .clamp(-self.limits, self.limits))
        data.ctrl[:, self.aids] = self.torque
