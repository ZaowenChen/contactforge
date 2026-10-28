"""Fixed convex profiles, chamfer-preserving collision meshes and exact shaft checks."""
import math
import mujoco
import numpy as np
import torch
from scipy.spatial import ConvexHull
from mjlab_contact_prep.hole_collision import clip

LEVELS = np.array([0., -.001, -.0395, -.04])
CHAMFERS = np.array([.001, 0., 0., .0005])
SHAFT_LENGTH = .05


def profile(shape, width=.0346):
    n = {"square": 4, "hexagon": 6}[shape]
    phase = math.pi/4 if n == 4 else 0.
    angles = phase + np.arange(n)*2*math.pi/n
    vertices = width/2/math.cos(math.pi/n)*np.c_[np.cos(angles), np.sin(angles)]
    normals = np.c_[np.cos(angles+math.pi/n), np.sin(angles+math.pi/n)]
    return vertices, normals


def perimeter(vertices, count=64):
    # Include every corner, then distribute the remaining samples along edges.
    n = len(vertices)
    counts = np.full(n, count//n)
    counts[:count % n] += 1
    return np.concatenate([p[None]+np.arange(k)[:,None]/k*(q-p)[None]
                           for p,q,k in zip(vertices,np.roll(vertices,-1,axis=0),counts)])


def add_convex(spec, body, name, vertices, **attrs):
    v = np.unique(np.round(vertices,14),axis=0)
    hull = ConvexHull(v)
    faces = hull.simplices.copy()
    for i,f in enumerate(faces):
        if np.dot(np.cross(v[f[1]]-v[f[0]],v[f[2]]-v[f[0]]),hull.equations[i,:3]) < 0:
            faces[i] = f[[0,2,1]]
    spec.add_mesh(name=name,uservert=v.ravel().tolist(),userface=faces.ravel().tolist())
    body.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,**attrs)


def replace_shaft(robot, cfg):
    shape, = cfg.assembly_shapes
    vertices,_ = profile(shape,cfg.polygon_peg_width_m)
    # Preserve the existing shaft's total mass, rigid mounting and connector.
    source_mass = float(robot.compile().body("peg_pa2").mass[0])
    for geom in list(robot.body("peg_pa2").geoms):
        robot.delete(geom)
    if cfg.wrist_compliance_enabled:
        # Match the frozen rigid pose while placing the physical shaft downstream
        # of the passive carrier, whose local frame has a 90-degree yaw.
        shaft = robot.body("peg_root").add_body(name="polygon_shaft",pos=(.017625,.017625,0.),
                                               quat=(math.sqrt(.5),0.,0.,-math.sqrt(.5)))
    else:
        shaft = robot.body("hand_mount").add_body(name="polygon_shaft",pos=(.019625,.004625,.238))
    v = np.concatenate([np.c_[vertices,np.full(len(vertices),z)] for z in (-SHAFT_LENGTH,0.)])
    add_convex(robot,shaft,"polygon_shaft",v,mass=source_mass,
               contype=1,conaffinity=1,condim=3,friction=(.9,.005,.0001),
               solref=(.002,1.),solimp=(.99,.999,.001,.5,2.),rgba=(.65,.7,.76,1.))


def polygon_hole_spec(cfg):
    shape, = cfg.assembly_shapes
    _, normals = profile(shape,cfg.polygon_peg_width_m)
    apothem = cfg.polygon_peg_width_m/2+cfg.clearance_per_side_m
    s = mujoco.MjSpec()
    body = s.worldbody.add_body(name="root_hole",mocap=True,pos=(.8663574,-.0017257,.1))
    body.add_site(name="hole_mouth_site",size=(.001,)*3,rgba=(1,.1,.8,1))
    attrs = dict(contype=1,conaffinity=1,condim=3,friction=(.9,.005,.0001),
                 solref=(.002,1.),solimp=(.99,.999,.001,.5,2.),mass=0.,
                 rgba=(.48,.52,.57,1.),margin=0.)
    for i,normal in enumerate(normals):
        angle = math.atan2(normal[1],normal[0]);half = math.pi/len(normals)
        low = np.array([-math.sin(angle-half),math.cos(angle-half)])
        high = np.array([math.sin(angle+half),-math.cos(angle+half)])
        # Separate convex slabs preserve the entry/exit chamfers exactly.
        for slab in range(3):
            vertices=[]
            for level in (slab,slab+1):
                poly=np.array([[-.045,-.045],[.045,-.045],[.045,.045],[-.045,.045]])
                for n,b in ((low,0.),(high,0.),(normal,apothem+CHAMFERS[level])):
                    poly=clip(poly,n,b)
                vertices.extend(np.c_[poly,np.full(len(poly),LEVELS[level])])
            add_convex(s,body,f"polygon_wall_{i}_{slab}",vertices,**attrs)
    return s


class PolygonGeometry:
    def __init__(self, shape, width, clearance, device):
        v,n = profile(shape,width)
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
        self.vertices=tensor(v);self.normals=tensor(n)
        self.apothem=width/2+clearance
        self.mouth_vertices=tensor(v*((self.apothem+.001)/(width/2)))
        edge=perimeter(v)
        self.points=tensor(np.c_[edge,np.zeros(len(edge))])
        self.prism=tensor(np.concatenate([np.c_[v,np.full(len(v),z)] for z in (0.,-SHAFT_LENGTH)]))
        count=len(v);idx=np.arange(count)
        edges=np.concatenate([np.c_[idx,(idx+1)%count],
                              np.c_[idx+count,(idx+1)%count+count],np.c_[idx,idx+count]])
        self.edges=torch.as_tensor(edges,dtype=torch.long,device=device)
        self.levels=tensor(LEVELS)
        self.period=2*math.pi/count

    def inside(self, xy, chamfer=0.):
        return (xy @ self.normals.T <= self.apothem+chamfer+1e-9).all(-1)

    def signed_mouth_distance(self, xy):
        a=self.mouth_vertices;b=a.roll(-1,0);edge=b-a
        delta=xy[...,None,:]-a
        t=(delta*edge).sum(-1)/edge.square().sum(-1)
        distance=(delta-t.clamp(0,1)[...,None]*edge).norm(dim=-1).min(-1).values
        return torch.where(self.inside(xy,.001),-distance,distance)

    def shaft_error(self, center, rotation):
        """Max half-space violation over the shaft clipped by every bore slab.

        Clipped convex polytope vertices are original prism vertices or edge /
        slab-plane intersections. Checking this complete set is sufficient for
        each linear wall inequality, including tilted edges and both end faces.
        """
        vertices=center[:,None]+self.prism[None]@rotation.transpose(1,2)
        a=vertices[:,self.edges[:,0]];b=vertices[:,self.edges[:,1]]
        dz=b[...,2]-a[...,2]
        valid_dz=dz.abs()>1e-9
        t=(self.levels[None,None]-a[...,2,None])/torch.where(valid_dz,dz,1.)[...,None]
        intersections=a[:,:,None]+t[...,None]*(b-a)[:,:,None]
        valid=valid_dz[...,None] & (t>=0) & (t<=1)
        candidates=torch.cat((vertices,intersections.flatten(1,2)),1)
        z=candidates[...,2]
        mask=torch.cat(((vertices[...,2]<=0)&(vertices[...,2]>=-.04),valid.flatten(1,2)),1)
        # Exact continuous linear chamfers; straight section between them.
        allowance=self.apothem+(z+.001).clamp_min(0)+(-.0395-z).clamp_min(0)
        violation=(candidates[...,:2]@self.normals.T).max(-1).values-allowance
        return torch.where(mask,violation,-torch.ones_like(violation)).max(-1).values

    def events(self, points, rotation, error, cfg):
        height=points[...,2]
        legal=(error<=cfg.geometry_tolerance_m)&(rotation[:,2,2]<0)&self.inside(points[...,:2]).all(-1)
        captured=legal&(height.max(-1).values<=-cfg.capture_depth_m)
        seated=legal&((height+cfg.goal_depth_m).abs()<=cfg.seating_tolerance_m).all(-1)
        return captured,seated

    def yaw_error(self, rotation):
        angle=torch.atan2(rotation[:,1,0],rotation[:,0,0])
        return torch.remainder(angle+self.period/2,self.period)-self.period/2


def audit_assets(destination,cfg):
    from dataclasses import replace
    from pathlib import Path
    import json
    import hashlib
    from .assets import assembly_spec, generate_circle_assets
    folder=Path(destination);folder.mkdir(parents=True,exist_ok=False)
    reports={}
    for shape in cfg.assembly_shapes:
        selected=replace(cfg,assembly_shapes=[shape])
        if shape=="circle":
            reports[shape]=generate_circle_assets(folder/shape,selected)
            assembly=assembly_spec(selected).compile()
            path=folder/f"{shape}.mjb";mujoco.mj_saveModel(assembly,str(path))
            reports[shape].update(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  symmetry_period_deg=None)
            continue
        spec=polygon_hole_spec(selected);model=spec.compile();data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        center=data.site("hole_mouth_site").xpos.copy()
        _,normals=profile(shape,cfg.polygon_peg_width_m)
        apothem=cfg.polygon_peg_width_m/2+cfg.clearance_per_side_m
        errors=[];face_gaps=[]
        for z in (-.0005,-.001,-.02,-.0395,-.03975):
            allowance=apothem+max(z+.001,0)+max(-.0395-z,0)
            # Avoid zero-area shared triangle edges; test both sides of every
            # polygon corner separately to verify the sectors meet there.
            corners=np.arctan2(normals[:,1],normals[:,0])+math.pi/len(normals)
            angles=np.r_[np.linspace(0,2*np.pi,720,endpoint=False)+1e-5,
                         corners-1e-5,corners+1e-5]
            directions=np.r_[np.c_[np.cos(angles),np.sin(angles)],normals]
            for direction in directions:
                ray=mujoco.mj_ray(model,data,center+[0,0,z],np.r_[direction,0.],
                                 None,1,-1,np.zeros(1,dtype=np.int32))
                expected=allowance/(normals@direction).max()
                errors.append(abs(ray-expected))
            if z==-.02:
                for direction in normals:
                    ray=mujoco.mj_ray(model,data,center+[0,0,z],np.r_[direction,0.],
                                     None,1,-1,np.zeros(1,dtype=np.int32))
                    face_gaps.append(ray-cfg.polygon_peg_width_m/2)
        assert max(errors)<2e-7, (shape,max(errors))
        assert max(abs(np.array(face_gaps)-cfg.clearance_per_side_m))<2e-7
        assembly=assembly_spec(selected).compile()
        path=folder/f"{shape}.mjb";mujoco.mj_saveModel(assembly,str(path))
        reports[shape]=dict(peg_across_flats_m=cfg.polygon_peg_width_m,
            hole_across_flats_m=2*apothem,clearance_per_side_m=cfg.clearance_per_side_m,
            measured_normal_gaps_m=face_gaps,max_ray_error_m=max(errors),
            plate_depth_m=.04,shaft_length_m=SHAFT_LENGTH,entry_chamfer_m=.001,
            exit_chamfer_m=.0005,shaft_mass_kg=float(assembly.body("robot/polygon_shaft").mass[0]),
            mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            visual_collision_same_geometry=True,symmetry_period_deg=360/len(normals))
    (folder/"geometry.json").write_text(json.dumps(reports,indent=2)+"\n")
    return reports
