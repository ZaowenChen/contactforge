"""Disjoint convex partition of the existing 72-sided chamfered bore.

The original strips describe a circumscribed 35 mm polygonal bore, with
1 mm entrance and .5 mm exit chamfers. Intersect each strip with its own
5-degree angular cell so coplanar surfaces no longer overlap in area.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.spatial import ConvexHull
import mujoco

LEVELS=((0.,.0185),(-.001,.0175),(-.0395,.0175),(-.04,.018))
OUTER_RECTS=((- .044,.044,.020,.045),(-.044,.044,-.045,-.020),(.020,.045,-.044,.044),(-.045,-.020,-.044,.044))

def clip(poly,normal,offset):
    result=[]
    for p,q in zip(poly,np.roll(poly,-1,axis=0)):
        a=float(p@normal-offset);b=float(q@normal-offset)
        if a>=-1e-14:result.append(p)
        if (a>0 and b<0) or (a<0 and b>0):result.append(p+(q-p)*a/(a-b))
    return np.asarray(result)

@lru_cache(maxsize=1)
def partition():
    sectors=[]
    for i in range(72):
        theta=math.radians(5*i);half=math.radians(2.5)
        n=np.array([math.cos(theta),math.sin(theta)])
        lo=np.array([-math.sin(theta-half),math.cos(theta-half)])
        hi=np.array([math.sin(theta+half),-math.cos(theta+half)])
        vertices=[]
        for z,r in LEVELS:
            poly=np.array([[-.02,-.02],[.02,-.02],[.02,.02],[-.02,.02]])
            for normal,offset in [(lo,0),(hi,0),(n,r)]:poly=clip(poly,normal,offset)
            vertices.extend(np.c_[poly,np.full(len(poly),z)])
        v=np.unique(np.round(vertices,14),axis=0);h=ConvexHull(v)
        faces=h.simplices.copy()
        for j,f in enumerate(faces):
            if np.dot(np.cross(v[f[1]]-v[f[0]],v[f[2]]-v[f[0]]),h.equations[j,:3])<0:faces[j]=f[[0,2,1]]
        sectors.append((v,faces))
    boxes=[];grid=[-.045,-.044,-.02,.02,.044,.045]
    for x0,x1 in zip(grid,grid[1:]):
        for y0,y1 in zip(grid,grid[1:]):
            x=(x0+x1)/2;y=(y0+y1)/2
            if any(a<x<b and c<y<d for a,b,c,d in OUTER_RECTS):
                boxes.append(((x,y,-.02),((x1-x0)/2,(y1-y0)/2,.02)))
    return sectors,boxes

def replace_collision(spec):
    # Original visuals and body/site structure are kept for old state replay.
    for geom in spec.geoms:
        geom.contype=0;geom.conaffinity=0
    root=spec.body('root_hole');sectors,boxes=partition()
    attrs=dict(condim=4,contype=1,conaffinity=1,friction=(.9,.2,.2),solref=(.001,2.),rgba=(0,0,0,0),group=3,mass=0)
    for i,(v,f) in enumerate(sectors):
        name=f'partition_sector_{i:02d}'
        spec.add_mesh(name=name,uservert=v.ravel().tolist(),userface=f.ravel().tolist())
        root.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,**attrs)
    for i,(pos,size) in enumerate(boxes):
        root.add_geom(name=f'partition_outer_{i:02d}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=pos,size=size,**attrs)
