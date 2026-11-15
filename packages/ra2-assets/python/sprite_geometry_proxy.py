#!/usr/bin/env python3
"""Geometry-only orthographic sprite proxies. No colour is generated.

JSON supplies camera basis, scale, screen origin and boxes/wedges/cylinders/spheres
in world coordinates. Outputs world normals, world height, plane IDs and a sampled
screen-space hemisphere AO estimate; invisible hemisphere samples cannot occlude.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from sprite_normals import CAM_RIGHT, CAM_UP, CAM_TOWARDS, normalise


def mesh(p):
    c=np.array(p['center'],float);d=np.array(p['size'],float);kind=p['kind']
    if kind in ('box','wedge'):
        v=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])*d+c
        if kind=='wedge':
            # Upper surface slopes along a chosen horizontal axis; lower surface stays flat.
            axis=p.get('slope_axis',2);top=v[:,1]>c[1];v[top,1]+=p.get('slope',.3)*(v[top,axis]-c[axis])
        quads=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]
        faces=[(q[0],q[i],q[i+1]) for q in quads for i in (1,2)]
    elif kind=='sphere':
        v=[];faces=[];ns=24;nr=16
        for j in range(nr+1):
            theta=np.pi*j/nr
            for i in range(ns):
                a=2*np.pi*i/ns;v.append(c+d*.5*np.array([np.sin(theta)*np.cos(a),np.cos(theta),np.sin(theta)*np.sin(a)]))
        for j in range(nr):
            for i in range(ns):
                a=j*ns+i;b=j*ns+(i+1)%ns;faces.extend([(a,b,a+ns),(b,b+ns,a+ns)])
        v=np.array(v)
    elif kind=='cylinder':
        axis=p.get('axis',0);cross=[i for i in range(3) if i!=axis];v=[];faces=[];ns=20
        for end in (-.5,.5):
            for i in range(ns):
                a=2*np.pi*i/ns;q=c.copy();q[axis]+=end*d[axis];q[cross]+=d[cross]*.5*np.array([np.cos(a),np.sin(a)]);v.append(q)
        v.extend([c-np.eye(3)[axis]*d[axis]/2,c+np.eye(3)[axis]*d[axis]/2]);v=np.array(v)
        for i in range(ns):
            j=(i+1)%ns;faces.extend([(i,j,i+ns),(j,j+ns,i+ns),(2*ns,j,i),(2*ns+1,i+ns,j+ns)])
    else:raise ValueError(kind)
    return v,faces


def project(v, config):
    return np.stack([v@CAM_RIGHT,-v@CAM_UP],-1)*config['pixels_per_world']+config['origin']


def silhouette(config):
    """Convex primitive coverage at pixel centers, matching the material rasterizer.

    Rounding vertices into an inclusive fill polygon overstates IoU for small sprites.
    The half-plane test uses the same (x+.5,y+.5) sample convention as rasterize.
    """
    w,h=config['size'];mask=np.zeros((h,w),bool)
    for p in config['primitives']:
        v,_=mesh(p);s=project(v,config)
        hull=cv2.convexHull(s.astype(np.float32)).reshape(-1,2)
        x0,y0=np.maximum(np.floor(hull.min(0)).astype(int),0)
        x1,y1=np.minimum(np.ceil(hull.max(0)).astype(int),[w-1,h-1])
        if x1<x0 or y1<y0:continue
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1];q=np.stack([xx+.5,yy+.5],-1)
        edge=np.roll(hull,-1,axis=0)-hull
        rel=q[...,None,:]-hull
        cross=edge[:,0]*rel[...,1]-edge[:,1]*rel[...,0]
        inside=(cross>=-1e-5).all(-1)|(cross<=1e-5).all(-1)
        mask[y0:y1+1,x0:x1+1]|=inside
    return mask


def rasterize(config):
    w,h=config['size'];depth=np.full((h,w),-1e6);normal=np.zeros((h,w,3));position=np.zeros((h,w,3));ids=np.zeros((h,w),np.uint16)
    for pi,p in enumerate(config['primitives']):
        verts,faces=mesh(p);screen=project(verts,config)
        for fi,f in enumerate(faces):
            v=verts[list(f)];s=screen[list(f)];n=normalise(np.cross(v[1]-v[0],v[2]-v[0]))
            if n@((v.mean(0))-np.array(p['center']))<0:n=-n
            if n@CAM_TOWARDS<=0:continue
            x0,y0=np.maximum(np.floor(s.min(0)).astype(int),0);x1,y1=np.minimum(np.ceil(s.max(0)).astype(int),[w-1,h-1])
            if x1<x0 or y1<y0:continue
            yy,xx=np.mgrid[y0:y1+1,x0:x1+1];q=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)
            mat=np.concatenate([s,np.ones((3,1))],1).T
            if abs(np.linalg.det(mat))<1e-8:continue
            bary=q@np.linalg.inv(mat).T;world=bary@v;z=world@CAM_TOWARDS
            ok=(bary.min(-1)>=-1e-6)&(z>depth[y0:y1+1,x0:x1+1])
            depth[y0:y1+1,x0:x1+1][ok]=z[ok];position[y0:y1+1,x0:x1+1][ok]=world[ok]
            nn=normalise((world-np.array(p['center']))/np.maximum(np.array(p['size'])**2,1e-5)) if p['kind']=='sphere' else np.broadcast_to(n,world.shape)
            normal[y0:y1+1,x0:x1+1][ok]=nn[ok];ids[y0:y1+1,x0:x1+1][ok]=pi*100+(fi//2+1 if p['kind'] in ('box','wedge') else fi+1)
    valid=ids>0;occlusion=np.zeros((h,w));samples=0
    yy,xx=np.indices(valid.shape)
    # Sample world hemisphere directions at two radii, project and depth-test against the proxy.
    for direction in ([1,1,0],[-1,1,0],[0,1,1],[0,1,-1],[1,1,1],[-1,1,1],[1,1,-1],[-1,1,-1]):
        direction=normalise(np.array(direction,float))
        for radius in (.08,.2):
            query=position+direction*radius;screen=project(query,config)
            sx=np.clip(screen[...,0].astype(int),0,w-1);sy=np.clip(screen[...,1].astype(int),0,h-1)
            above=(normal@direction)>0;blocked=depth[sy,sx]>(query@CAM_TOWARDS+.025)
            occlusion+=above&blocked&valid;samples+=1
    ao=np.clip(1-.6*occlusion/samples,.3,1);ao[~valid]=0
    return dict(normal=normal,height=position[...,1],ao=ao,plane_ids=ids,mask=valid,depth=depth)


def visible_plane_edges(config):
    """Fast painter projection used only by fitting; final buffers use the exact z rasterizer."""
    w,h=config['size'];planes=np.zeros((h,w),np.int32);triangles=[]
    for pi,p in enumerate(config['primitives']):
        v,faces=mesh(p);screen=project(v,config)
        for fi,f in enumerate(faces):
            vv=v[list(f)];n=normalise(np.cross(vv[1]-vv[0],vv[2]-vv[0]))
            if n@(vv.mean(0)-np.array(p['center']))<0:n=-n
            if n@CAM_TOWARDS<=0:continue
            group=fi//2+1 if p['kind'] in ('box','wedge') else 1
            triangles.append((vv.mean(0)@CAM_TOWARDS,screen[list(f)],1+pi*10+group))
    for _,triangle,plane in sorted(triangles,key=lambda t:t[0]):
        cv2.fillConvexPoly(planes,np.rint(triangle).astype(np.int32),plane)
    edges=np.zeros((h,w),bool);edges[1:]|=planes[1:]!=planes[:-1];edges[:,1:]|=planes[:,1:]!=planes[:,:-1]
    return edges

def fit(config,target,rounds=4,target_edges=None,deadline=None):
    """Bounded coordinate search of primitive centers/extents against original alpha and edges."""
    import time
    def score():
        m=silhouette(config);iou=(m&target).sum()/max(1,(m|target).sum())
        # A small edge objective discourages a proxy filling original interior holes.
        a=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        b=cv2.morphologyEx(target.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        value=iou+.003*(a.astype(bool)&b.astype(bool)).sum()/max(1,(a.astype(bool)|b.astype(bool)).sum())
        if target_edges is not None:
            proxy=visible_plane_edges(config)&target
            near=cv2.dilate(target_edges.astype(np.uint8),np.ones((3,3),np.uint8))>0
            value+=.003*(proxy&near).sum()/max(1,proxy.sum())
        return value,iou
    best,iou=score();history=[iou]
    for step in [.08,.04,.02,.01][:rounds]:
        for p in config['primitives']:
            if deadline is not None and time.time()>=deadline:return history+[iou]
            for field in ('center','size'):
                for axis in range(3):
                    old=p[field][axis];chosen=old
                    for sign in (-1,1):
                        p[field][axis]=old+sign*step
                        limits=p.get('bounds',{}).get(field)
                        if (field=='size' and p[field][axis]<.03) or (limits is not None and not limits[axis][0]<=p[field][axis]<=limits[axis][1]):continue
                        value,ii=score()
                        if value>best:best=value;chosen=p[field][axis];iou=ii
                    p[field][axis]=chosen
        history.append(iou)
    return history


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    c=json.loads(a.config.read_text());buffers=rasterize(c);np.savez_compressed(a.output,**buffers)
if __name__=='__main__':main()
