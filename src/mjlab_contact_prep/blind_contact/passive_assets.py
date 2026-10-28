"""Opt-in simulated wrist compliance; the source XML assets are never edited."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

COMPLIANCE_JOINTS = ("peg_comp_x", "peg_comp_y", "peg_comp_z")


def configure_arm_dynamics(robot, cfg):
    for index in range(1,7):
        joint = robot.joint(f"joint_{index}")
        joint.frictionloss *= cfg.joint_frictionloss_scale
        joint.armature *= cfg.joint_armature_scale
    if cfg.arm_gravity_compensation:
        # A MuJoCo force sensor measures the child-to-parent body interaction.
        # Preserve ALL masses below that boundary, including the hand/peg load.
        model = robot.compile()
        body_id = int(model.body_parentid[model.body("hand_mount").id])
        while body_id:
            robot.body(model.body(body_id).name).gravcomp = 1.0
            body_id = int(model.body_parentid[body_id])


def add_wrist_compliance(robot, cfg):
    if not cfg.wrist_compliance_enabled:
        return
    model = robot.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    carrier = robot.body("peg_root")
    body_id = model.body("peg_root").id
    carrier_rotation = data.xmat[body_id].reshape(3,3)
    tip = robot.site("peg_geometry_tip_site")
    robot.body("hand_mount").add_site(name="nominal_peg_tip_site",pos=tip.pos.copy(),
        quat=tip.quat.copy(),size=(.0005,)*3,rgba=(0.,0.,0.,0.))
    for name in ("peg_geometry_tip_site","peg_axis_visual"):
        site = robot.site(name)
        sid = model.site(name).id
        position = carrier_rotation.T@(data.site_xpos[sid]-data.xpos[body_id])
        rotation = carrier_rotation.T@data.site_xmat[sid].reshape(3,3)
        quat = Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        attrs = dict(type=site.type,size=site.size.copy(),rgba=site.rgba.copy(),group=site.group)
        robot.delete(site)
        carrier.add_site(name=name,pos=position,quat=quat,**attrs)
    # A joint-bearing carrier requires positive own inertia; this explicit
    # 20 g adapter is included in sensor payload compensation and the audit.
    carrier.explicitinertial = True
    carrier.mass = .02
    carrier.inertia[:] = (2e-5,2e-5,2e-5)
    carrier.gravcomp = 0.0
    for index,name in enumerate(COMPLIANCE_JOINTS):
        axial = index == 2
        prefix = "wrist_compliance_z_" if axial else "wrist_compliance_xy_"
        travel = getattr(cfg,prefix+"travel_m")
        carrier.add_joint(name=name,type=mujoco.mjtJoint.mjJNT_SLIDE,axis=np.eye(3)[index],
            limited=True,range=(-travel,travel),stiffness=getattr(cfg,prefix+"stiffness_n_per_m"),
            damping=getattr(cfg,prefix+"damping_ns_per_m"),armature=.005,frictionloss=0.,
            ref=0.,springref=0.)


def passive_indices(model):
    qids=[];vids=[]
    for name in COMPLIANCE_JOINTS:
        joint=model.joint("robot/"+name)
        qids.append(int(joint.qposadr[0]));vids.append(int(joint.dofadr[0]))
    return qids,vids
