"""Fixed circle/polygon batches, one shared policy and one pooled PPO update."""
from dataclasses import replace
import numpy as np
import torch
from .study_environment import StudyInsertionEnv, edge_labels
from .polygon_geometry import PolygonGeometry, profile, perimeter
from .assets import peg_geometry

ALIGNED_ROTATION=np.diag([-1.,1.,-1.])


class CircleStudyEnv(StudyInsertionEnv):
    """The existing circular asset with the same nominal pose as the polygons."""
    nominal_rotation=ALIGNED_ROTATION

    def _make_reset_bank(self):
        super()._make_reset_bank()
        self.shape="circle"
        self.reset_audit.update(shape=self.shape,peg_diameter_m=2*peg_geometry()["radius_m"],
                                clearance_per_side_m=self.cfg.clearance_per_side_m)

    def step(self, action):
        obs,reward,done,info=super().step(action)
        # Placeholder for concatenation only. Reports must honor the mask:
        # an axially symmetric circular fit has no yaw alignment target.
        info["yaw_error_rad"]=torch.zeros(self.num_envs,device=self.device)
        info["yaw_alignment_required"]=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        return obs,reward,done,info


def make_shape_environment(cfg, **kwargs):
    shape,=cfg.assembly_shapes
    cls=CircleStudyEnv if shape=="circle" else PolygonStudyEnv
    return cls(cfg,**kwargs)


class PolygonStudyEnv(StudyInsertionEnv):
    # 180 degrees is an equivalent alignment for both profiles and stays near
    # the source wrist yaw (174.55 degrees), avoiding a new robot IK branch.
    nominal_rotation=ALIGNED_ROTATION

    def _make_reset_bank(self):
        self.shape,=self.cfg.assembly_shapes
        self.geometry=PolygonGeometry(self.shape,self.cfg.polygon_peg_width_m,
                                      self.cfg.clearance_per_side_m,self.device)
        super()._make_reset_bank()
        self.reset_audit.update(shape=self.shape,peg_across_flats_m=self.cfg.polygon_peg_width_m,
                                clearance_per_side_m=self.cfg.clearance_per_side_m)

    def reset_outline(self):
        v,_=profile(self.shape,self.cfg.polygon_peg_width_m)
        # Dense edge samples improve contact-start acceptance on tilted faces.
        xy=perimeter(v,256)
        return np.c_[xy,np.zeros(len(xy))]

    def reset_outside(self, xy):
        _,n=profile(self.shape,self.cfg.polygon_peg_width_m)
        return (xy@n.T>self.cfg.polygon_peg_width_m/2+self.cfg.clearance_per_side_m+.001).any(-1)

    def edge_points(self):
        return self.geometry.points

    def inside_bore(self, points):
        return self.geometry.inside(points[...,:2])

    def events(self, points, rotation, shaft_error):
        return self.geometry.events(points,rotation,shaft_error,self.cfg)

    def truth(self):
        points,center,rotation=self.local_geometry()
        labels=edge_labels(self.geometry.signed_mouth_distance(points[...,:2]),
                           points[...,2],self.cfg.contact_band_m)
        error=self.geometry.shaft_error(center,rotation)
        return labels,-points[...,2].max(-1).values,error,center,rotation[:,:,2]

    def step(self, action):
        # Record terminal yaw before the parent's automatic reset.
        auto=self.auto_reset;self.auto_reset=False
        try:
            obs,reward,done,info=super().step(action)
        finally:
            self.auto_reset=auto
        info["yaw_error_rad"]=self.geometry.yaw_error(self.local_geometry()[2])
        info["yaw_alignment_required"]=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        if auto:
            self.reset(done.nonzero().flatten())
            obs=self.observations()
        return obs,reward,done,info


class _ReadOnlyData:
    def __init__(self, env):
        self.env=env

    def __getattr__(self, name):
        if name not in ("qpos","qvel","mocap_pos","mocap_quat"):
            raise AttributeError(name)
        # SimData fields are proxies; materialize before concatenating a list.
        return torch.cat([getattr(e.data,name).clone() for e in self.env.envs])


class MixedStudyEnv:
    def __init__(self,cfg,device="cuda:0",auto_reset=True):
        self.cfg=cfg.validate();self.device=device;self.num_envs=cfg.num_envs
        self.shapes=cfg.assembly_shapes;self.auto_reset=auto_reset
        size=cfg.num_envs//len(self.shapes)
        self.envs=[make_shape_environment(replace(cfg,assembly_shapes=[shape],num_envs=size,
                   reset_bank_size=cfg.reset_bank_size//len(self.shapes),seed=cfg.seed+1000*i),
                   device=device,auto_reset=auto_reset) for i,shape in enumerate(self.shapes)]
        self.shape_ids=torch.arange(len(self.shapes),device=device).repeat_interleave(size)
        vocabulary=cfg.shape_vocabulary
        code_ids=torch.tensor([vocabulary.index(s) for s in self.shapes],device=device).repeat_interleave(size)
        self.shape_code=torch.nn.functional.one_hot(code_ids,len(vocabulary)).float()
        self.critic_dim=235+len(vocabulary)
        self.data=_ReadOnlyData(self)
        self.reset_audit=dict(size=cfg.reset_bank_size,all_starts_have_actual_contact=all(
            e.reset_audit["all_starts_have_actual_contact"] for e in self.envs),
            by_shape={s:e.reset_audit for s,e in zip(self.shapes,self.envs)},
            critic_shape_vocabulary=list(vocabulary))

    def __getattr__(self,name):
        if name in ("history","active","peak_force","reset_radius","captured","fault_code"):
            return torch.cat([getattr(e,name) for e in self.envs])
        if name in ("max_contacts","max_constraints","buffer_overflow"):
            values=torch.stack([getattr(e,name) for e in self.envs])
            return values.any(0) if name=="buffer_overflow" else values.max(0).values
        raise AttributeError(name)

    def combine_observations(self, observations):
        obs={k:torch.cat([o[k] for o in observations]) for k in observations[0]}
        obs["critic"]=torch.cat((obs["critic"],self.shape_code),-1)
        return obs

    def observations(self):
        return self.combine_observations([e.observations() for e in self.envs])

    def step(self, actions):
        results=[e.step(a) for e,a in zip(self.envs,actions.chunk(len(self.envs)))]
        obs=self.combine_observations([r[0] for r in results])
        info={k:torch.cat([r[3][k] for r in results]) for k in results[0][3] if k!="terminal"}
        info["terminal"]=self.combine_observations([r[3]["terminal"] for r in results])
        info["shape_id"]=self.shape_ids
        return obs,torch.cat([r[1] for r in results]),torch.cat([r[2] for r in results]),info

    def reset(self,ids=None):
        size=self.envs[0].num_envs
        for i,e in enumerate(self.envs):
            selected=None if ids is None else ids[(ids>=i*size)&(ids<(i+1)*size)]-i*size
            e.reset(selected)
        return self.observations()

    def snapshot(self):
        return dict(shapes=self.shapes,groups=[e.snapshot() for e in self.envs])

    def restore_snapshot(self,state):
        if state["shapes"]!=self.shapes or len(state["groups"])!=len(self.envs):
            raise ValueError("snapshot geometry groups differ")
        for e,s in zip(self.envs,state["groups"]):
            e.restore_snapshot(s)

    def close(self):
        for e in self.envs:
            e.close()
