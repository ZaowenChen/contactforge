"""Revision 6: tool-frame Cartesian tip reference with a bounded F/T governor.

The policy moves a persistent tip reference (translation and rotation about the
nominal peg tip, expressed in the tool frame). Every physics step the governor
bounds the reference lead, removes motion that pushes against an excessive
measured load, retreats along it, and maps the remaining pose error through a
damped least-squares Jacobian solve into the existing joint impedance target.

Inputs are encoder kinematics and the compensated wrist F/T only; hole pose,
contact labels and the passive carrier state never enter this controller.
Thresholds are development settings, not certified physical force guarantees.
"""
import torch
from .control import JointImpedance


def skew(w):
    o = torch.zeros_like(w[..., 0])
    return torch.stack((torch.stack((o, -w[..., 2], w[..., 1]), -1),
                        torch.stack((w[..., 2], o, -w[..., 0]), -1),
                        torch.stack((-w[..., 1], w[..., 0], o), -1)), -2)


def so3_exp(w):
    theta = w.norm(dim=-1, keepdim=True)[..., None]
    small = theta < 1e-6
    safe = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, torch.ones_like(theta), torch.sin(safe)/safe)
    b = torch.where(small, torch.full_like(theta, .5), (1-torch.cos(safe))/safe.square())
    k = skew(w)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(k)
    return eye + a*k + b*(k@k)


def so3_log(r):
    v = .5*torch.stack((r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0],
                        r[..., 1, 0]-r[..., 0, 1]), -1)
    s = v.norm(dim=-1, keepdim=True)
    c = ((r[..., 0, 0]+r[..., 1, 1]+r[..., 2, 2]-1)/2).clamp(-1., 1.)[..., None]
    theta = torch.atan2(s, c)
    return v*torch.where(s > 1e-9, theta/s.clamp_min(1e-12), torch.ones_like(s))


def norm_limit(vector, limit):
    return vector*(limit/vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)


def comply(lead, load, threshold, band, step):
    """Per-axis deadzone admittance on a stored reference lead.

    ``load`` is the external load on the peg. Beyond ``threshold`` the lead
    retreats along the load by ``step`` per unit excess, and any lead component
    still pushing against the load is scaled to zero across ``band``.
    """
    magnitude = load.abs()
    excess = (magnitude-threshold).clamp_min(0.)
    direction = torch.sign(load)
    lead = lead + step*direction*excess
    pushing = lead*direction < 0
    scale = (1-excess/band).clamp(0., 1.)
    return torch.where(pushing, lead*scale, lead)


class CartesianTipImpedance(JointImpedance):
    def __init__(self, cfg, model, device, num_envs):
        super().__init__(cfg, model, device, num_envs)
        self.joint_ids = [model.joint(f"robot/joint_{i}").id for i in range(1, 7)]
        self.tip_id = model.site("robot/nominal_peg_tip_site").id
        self.ref_pos = torch.zeros(num_envs, 3, device=device)
        self.ref_rot = torch.eye(3, device=device).repeat(num_envs, 1, 1)
        self.eye6 = torch.eye(6, device=device)
        w = cfg.cart_rotation_weight_m
        self.weights = torch.tensor([1., 1., 1., w, w, w], device=device)
        dt = cfg.physics_dt
        self.force_threshold = torch.tensor([cfg.lateral_force_threshold_n]*2+[cfg.axial_force_threshold_n],
                                            device=device)
        self.force_step = cfg.force_retreat_m_per_ns*dt
        self.moment_step = cfg.moment_comply_rad_per_nms*dt

    def pose(self, data):
        return data.site_xpos[:, self.tip_id], data.site_xmat[:, self.tip_id].reshape(-1, 3, 3)

    def jacobian(self, data):
        axes = data.xaxis[:, self.joint_ids]
        arms = data.site_xpos[:, self.tip_id, None, :]-data.xanchor[:, self.joint_ids]
        return torch.cat((torch.linalg.cross(axes, arms, dim=-1).transpose(1, 2),
                          axes.transpose(1, 2)), dim=1)

    def reset(self, ids, data):
        p, r = self.pose(data)
        self.ref_pos[ids] = p[ids]
        self.ref_rot[ids] = r[ids]
        self.target[ids] = data.qpos[ids][:, self.qids]
        self.last_action[ids] = 0

    def set_action(self, data, action):
        mapped = action.tanh()
        self.last_action.copy_(mapped)
        # Leaky-anchored tool-frame lead prevents unbounded free-air integrator
        # windup while preserving smooth multi-step impedance compliance.
        p, r = self.pose(data)
        rt = r.transpose(1, 2)
        lead_p = .5*(rt@(self.ref_pos-p)[:, :, None]).squeeze(-1)+mapped[:, :3]*self.cfg.cart_translation_step_m
        lead_r = .5*(rt@so3_log(self.ref_rot@rt)[:, :, None]).squeeze(-1)+mapped[:, 3:]*self.cfg.cart_rotation_step_rad
        lead_p = norm_limit(lead_p, self.cfg.cart_translation_lead_m)
        lead_r = norm_limit(lead_r, self.cfg.cart_rotation_lead_rad)
        self.ref_pos.copy_(p+(r@lead_p[:, :, None]).squeeze(-1))
        self.ref_rot.copy_(so3_exp((r@lead_r[:, :, None]).squeeze(-1))@r)

    def cancel_reference(self, data, mask):
        """Cancel queued policy motion without teleporting the physical state."""
        p, r = self.pose(data)
        self.ref_pos[mask] = p[mask]
        self.ref_rot[mask] = r[mask]
        self.target[mask] = data.qpos[mask][:, self.qids]
        self.last_action[mask] = 0

    def govern(self, data, wrench_tool, stopped=None):
        cfg = self.cfg
        q = data.qpos[:, self.qids]
        p, r = self.pose(data)
        rt = r.transpose(1, 2)
        lead_p = (rt@(self.ref_pos-p)[:, :, None]).squeeze(-1)
        lead_r = (rt@so3_log(self.ref_rot@rt)[:, :, None]).squeeze(-1)
        if stopped is not None:
            # Ended worlds may continue simulating until their batch resets.
            # They receive only damping/load relief, never the old stored lead.
            lead_p = torch.where(stopped[:, None], 0., lead_p)
            lead_r = torch.where(stopped[:, None], 0., lead_r)
        retreat_p = norm_limit(
            lead_p + self.force_step*torch.sign(wrench_tool[:, :3])*(wrench_tool[:, :3].abs()-self.force_threshold).clamp_min(0.),
            cfg.cart_translation_lead_m)
        retreat_r = norm_limit(
            lead_r + self.moment_step*torch.sign(wrench_tool[:, 3:])*(wrench_tool[:, 3:].abs()-cfg.moment_threshold_nm).clamp_min(0.),
            cfg.cart_rotation_lead_rad)
        # Persist only the dt-scaled retreat so the 20 substeps do not compound scale**20.
        self.ref_pos.copy_(p+(r@retreat_p[:, :, None]).squeeze(-1))
        self.ref_rot.copy_(so3_exp((r@retreat_r[:, :, None]).squeeze(-1))@r)
        lead_p = norm_limit(comply(lead_p, wrench_tool[:, :3], self.force_threshold, cfg.force_band_n, self.force_step),
                            cfg.cart_translation_lead_m)
        lead_r = norm_limit(comply(lead_r, wrench_tool[:, 3:], cfg.moment_threshold_nm, cfg.moment_band_nm, self.moment_step),
                            cfg.cart_rotation_lead_rad)
        world_p = (r@lead_p[:, :, None]).squeeze(-1)
        world_r = (r@lead_r[:, :, None]).squeeze(-1)
        jac = self.jacobian(data)*self.weights[None, :, None]
        error = torch.cat((world_p, world_r), -1)*self.weights
        dq = torch.linalg.solve(jac.transpose(1, 2)@jac+cfg.cart_dls_damping**2*self.eye6,
                                (jac.transpose(1, 2)@error[:, :, None]).squeeze(-1))
        dq = dq.clamp(-cfg.action_scale, cfg.action_scale)
        self.target.copy_((q+dq).clamp(self.joint_ranges[:, 0], self.joint_ranges[:, 1]))
