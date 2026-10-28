"""Derive a larger bore without modifying the source robot or peg assets."""
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from mjlab_contact_prep.assets import ASSET_ROOT, START_ARM_QPOS, robot_spec
from mjlab_contact_prep.hole_collision import clip, partition
from .config import BlindConfig

PEG_CENTER_LOCAL = np.array([0.017625, 0.017625])
NOMINAL_TIP = np.array([0.8663574, -0.0017257, 0.1015])
# Preserve the source wrist yaw while removing its approximately 0.54 deg tilt.
NOMINAL_ROTATION = Rotation.from_euler("z", 174.55, degrees=True).as_matrix() @ np.diag([1., -1., -1.])


def peg_geometry():
    path = ASSET_ROOT / "peg/pa2.obj"
    vertices = np.array([[float(x) for x in line.split()[1:4]]
                         for line in path.read_text().splitlines() if line.startswith("v ")])
    radius = float(np.linalg.norm(vertices[:, :2] - PEG_CENTER_LOCAL, axis=1).max())
    return dict(radius_m=radius, shaft_length_m=float(np.ptp(vertices[:, 2])),
                center_local_m=PEG_CENTER_LOCAL.tolist(),
                source=str(path.relative_to(ASSET_ROOT.parent)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def bore_parts(clearance):
    radius = peg_geometry()["radius_m"] + clearance
    levels = [(0., radius + .001), (-.001, radius),
              (-.0395, radius), (-.04, radius + .0005)]
    parts = []
    for i in range(72):
        theta = 2 * math.pi * i / 72
        half = math.pi / 72
        normals = [(np.array([-math.sin(theta-half), math.cos(theta-half)]), 0.),
                   (np.array([math.sin(theta+half), -math.cos(theta+half)]), 0.)]
        wall = np.array([math.cos(theta), math.sin(theta)])
        vertices = []
        for z, r in levels:
            poly = np.array([[-.02,-.02],[.02,-.02],[.02,.02],[-.02,.02]])
            for normal, offset in normals + [(wall, r)]:
                poly = clip(poly, normal, offset)
            vertices.extend(np.c_[poly, np.full(len(poly), z)])
        v = np.unique(np.round(vertices, 14), axis=0)
        hull = ConvexHull(v)
        f = hull.simplices.copy()
        for j, face in enumerate(f):
            if np.dot(np.cross(v[face[1]]-v[face[0]], v[face[2]]-v[face[0]]),
                      hull.equations[j, :3]) < 0:
                f[j] = face[[0, 2, 1]]
        parts.append((v, f))
    return radius, levels, parts, partition()[1]


def hole_spec(clearance):
    s = mujoco.MjSpec()
    root = s.worldbody.add_body(name="root_hole", mocap=True, pos=(.8663574, -.0017257, .1))
    root.add_site(name="hole_mouth_site", size=(.001,)*3, rgba=(1, .1, .8, 1))
    radius, levels, sectors, boxes = bore_parts(clearance)
    # One geometry serves both rendering and collision: no old, narrow visual bore.
    attrs = dict(contype=1, conaffinity=1, condim=3, friction=(.9, .005, .0001),
                 solref=(.002, 1.), solimp=(.99, .999, .001, .5, 2.),
                 rgba=(.48, .52, .57, 1.), mass=0., margin=0.)
    for i, (v, f) in enumerate(sectors):
        name = f"bore_{i:02d}"
        s.add_mesh(name=name, uservert=v.ravel().tolist(), userface=f.ravel().tolist())
        root.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, **attrs)
    for i, (pos, size) in enumerate(boxes):
        root.add_geom(name=f"outer_{i:02d}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=pos, size=size, **attrs)
    return s


def assembly_spec(cfg: BlindConfig):
    robot = robot_spec()
    if cfg.wrist_compliance_enabled:
        from .passive_assets import configure_arm_dynamics
        configure_arm_dynamics(robot,cfg)
    polygon = cfg.study_revision >= 3 and cfg.assembly_shapes != ["circle"]
    if polygon:
        from .polygon_geometry import replace_shaft, polygon_hole_spec
        replace_shaft(robot, cfg)
    if cfg.wrist_compliance_enabled:
        from .passive_assets import add_wrist_compliance
        add_wrist_compliance(robot,cfg)
    robot.body("hand_mount").add_site(name="blind_tcp_site", pos=(0.,0.,0.13),
                                      size=(.0005,)*3, rgba=(0.,0.,0.,0.))
    for i, limit in enumerate(cfg.torque_limits, 1):
        a = robot.actuator(f"joint_{i}")
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.biastype = mujoco.mjtBias.mjBIAS_NONE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[:] = 0
        a.gainprm[0] = 1
        a.biasprm[:] = 0
        a.ctrlrange[:] = (-limit, limit)
        a.forcerange[:] = (-limit, limit)
    for name, sensor_type in [("tcp_linear_velocity", mujoco.mjtSensor.mjSENS_FRAMELINVEL),
                              ("tcp_angular_velocity", mujoco.mjtSensor.mjSENS_FRAMEANGVEL)]:
        robot.add_sensor(name=name, type=sensor_type, objtype=mujoco.mjtObj.mjOBJ_SITE,
                         objname="blind_tcp_site" if cfg.method == "study" else "peg_geometry_tip_site")
    s = mujoco.MjSpec()
    s.attach(robot, prefix="robot/", frame=s.worldbody.add_frame())
    hole = polygon_hole_spec(cfg) if polygon else hole_spec(cfg.clearance_per_side_m)
    s.attach(hole, prefix="hole/", frame=s.worldbody.add_frame())
    if cfg.method == "study":
        # Higher priority makes the randomized plate parameters govern the pair,
        # rather than MuJoCo taking the maximum with a fixed peg coefficient.
        for geom in s.geoms:
            if geom.name.startswith("hole/"):
                geom.priority = 1
    s.option.timestep = cfg.physics_dt
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    s.option.iterations = 50
    s.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    s.option.ccd_iterations = 50
    return s


def solve_ik(model, target_position, target_rotation=NOMINAL_ROTATION, initial=None):
    """CPU reset/calibration only. Never called to choose the policy's action."""
    d = mujoco.MjData(model)
    qids = np.array([model.joint(f"robot/joint_{i}").qposadr[0] for i in range(1,7)])
    vids = np.array([model.joint(f"robot/joint_{i}").dofadr[0] for i in range(1,7)])
    sid = model.site("robot/peg_geometry_tip_site").id
    d.qpos[qids] = START_ARM_QPOS if initial is None else initial
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(100):
        mujoco.mj_forward(model, d)
        rot = d.site_xmat[sid].reshape(3,3)
        error = np.r_[target_position-d.site_xpos[sid],
                      Rotation.from_matrix(target_rotation @ rot.T).as_rotvec()]
        if np.linalg.norm(error[:3]) < 1e-8 and np.linalg.norm(error[3:]) < 1e-7:
            return d.qpos.copy()
        mujoco.mj_jacSite(model, d, jp, jr, sid)
        jac = np.r_[jp[:, vids], jr[:, vids]]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6)*1e-8, error)
        d.qpos[qids] += np.clip(dq, -.05, .05)
    raise RuntimeError(f"reset IK did not converge: {error}")


def generate_assets(destination, cfg):
    """Save standalone derived-hole MJCF and audit actual compiled wall locations."""
    if cfg.study_revision >= 3:
        from .polygon_geometry import audit_assets
        return audit_assets(destination,cfg)
    return generate_circle_assets(destination,cfg)


def generate_circle_assets(destination, cfg):
    """Audit the existing circular shaft and its derived clearance bore."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    spec = hole_spec(cfg.clearance_per_side_m)
    (destination/"hole.xml").write_text(spec.to_xml())
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    center = data.site("hole_mouth_site").xpos
    distances = []
    for angle in np.linspace(0, 2*np.pi, 720, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle), 0.])
        origin = center + [0, 0, -.02]
        distance = mujoco.mj_ray(model, data, origin, direction, None, 1, -1, np.zeros(1, dtype=np.int32))
        if distance <= 0:
            raise AssertionError("missing bore wall")
        distances.append(distance)
    peg = peg_geometry()
    clearances = np.array(distances)-peg["radius_m"]
    if abs(clearances.min()-cfg.clearance_per_side_m) > 2e-7:
        raise AssertionError(f"wrong minimum clearance: {clearances.min()}")
    report = dict(peg=peg, clearance_per_side_m=cfg.clearance_per_side_m,
                  diameter_difference_m=2*cfg.clearance_per_side_m,
                  measured_radial_gap_min_m=float(clearances.min()),
                  measured_radial_gap_max_m=float(clearances.max()),
                  bore_apothem_m=peg["radius_m"]+cfg.clearance_per_side_m,
                  plate_depth_m=.04, entry_chamfer_m=.001, exit_chamfer_m=.0005,
                  ray_samples=720, visual_collision_same_geometry=True,
                  derived_xml_sha256=hashlib.sha256((destination/"hole.xml").read_bytes()).hexdigest())
    (destination/"geometry.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
