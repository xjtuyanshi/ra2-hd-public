"""Conservative material edits and connected-plane shading diagnostics.

These are art-supported hypotheses, not recovered ground truth. Callers inject exact
paint/emission masks, current material buffers and the renderer's baked-light vector.
"""
from __future__ import annotations
import cv2
import numpy as np
import sprite_normals as sn


def plane_regions(normal, valid, minimum=8):
    """Connected orientation regions from this variant's own quantized normals."""
    n=sn.normalise(normal)
    elevation=np.arcsin(np.clip(n[...,1],0,1))
    azimuth=np.arctan2(n[...,0],n[...,2])
    family=np.where(elevation>np.deg2rad(70),0,np.where(azimuth<np.pi/4,1,2))
    tilt=np.where(elevation>np.deg2rad(70),0,np.minimum(3,(elevation/np.deg2rad(22)).astype(int)))
    code=family*4+tilt
    ids=np.zeros(valid.shape,np.int32);next_id=1
    for value in np.unique(code[valid]):
        count,labels,stats,_=cv2.connectedComponentsWithStats(((code==value)&valid).astype(np.uint8),8)
        for k in range(1,count):
            if stats[k,4]<minimum:continue
            ids[labels==k]=next_id;next_id+=1
    return ids


def paint_planes(paint,valid):
    """Compact paint islands: tall projected quadrilaterals are wall/up plates.

    A roof diamond is about half as tall as wide at this camera. Tall islands have
    extra projected rise and are slope candidates. Filled convex support includes
    small original holes in the paint, without manual per-asset coordinates.
    """
    count,labels,stats,_=cv2.connectedComponentsWithStats((paint&valid).astype(np.uint8),8)
    slopes=np.zeros(valid.shape,np.int32);records=[];sid=0
    for k in range(1,count):
        x,y,w,h,area=stats[k]
        if area<max(16,.005*valid.sum()) or w<6 or h/w<.60 or area/(h*w)<.30:continue
        yy,xx=np.nonzero(labels==k);hull=cv2.convexHull(np.stack([xx,yy],-1).astype(np.int32))
        support=np.zeros(valid.shape,np.uint8);cv2.fillConvexPoly(support,hull,1)
        sid+=1;slopes[(support>0)&valid]=sid
        records.append(dict(id=sid,bbox=[int(x),int(y),int(w),int(h)],pixels=int(((support>0)&valid).sum()),source='exact paint-index component projected aspect and convex boundary'))
    return slopes,records


def orient_slopes(normal,rgb,valid,slope_ids,light,reference_normal=None):
    """Choose +x/+z slope family at region level using independent baked shading.

    Anchor means are the art luminance on large, near-vertical reference faces.
    Normalize chroma to grey max-channel intensity to compare paint with neutral
    metal; a confidence margin leaves ambiguous regions unchanged. Candidate tilt
    is preserved, and only the horizontal family changes. No per-pixel brightness
    to normal regression is performed.
    """
    result=normal.copy();ref=normal if reference_normal is None else reference_normal
    # Max-channel intensity retains the original house-ramp shading for saturated paint.
    lum=rgb.max(-1)
    labels=np.stack([ref[...,1],ref[...,2],ref[...,0]],-1).argmax(-1)
    means={}
    for f in (sn.WALL_Z,sn.WALL_X):
        m=valid&(labels==f)&(ref[...,1]<.45)&(slope_ids==0)
        means[f]=float(np.median(lum[m])) if m.sum()>=16 else None
    records=[]
    # Light, not coordinate signs, defines which wall family is the brighter one.
    bright=sn.WALL_Z if light[2]>light[0] else sn.WALL_X
    dark=sn.WALL_X if bright==sn.WALL_Z else sn.WALL_Z
    lo,hi=means.get(dark),means.get(bright)
    for k in np.unique(slope_ids[slope_ids>0]):
        m=(slope_ids==k)&valid
        if m.sum()<8:continue
        mean=float(np.median(lum[m]));prior=sn.normalise(result[m].mean(0))
        tilt=float(np.clip(prior[1],.25,.90));horizontal=np.sqrt(1-tilt*tilt)
        candidates=np.array([[0,tilt,horizontal],[horizontal,tilt,0]])
        prior_family=int(np.argmax([prior[2],prior[0]]));chosen=prior_family
        confident=lo is not None and hi is not None and hi>lo+.025
        if confident:
            relative=(mean-lo)/(hi-lo)
            if relative>.65:chosen=0 if bright==sn.WALL_Z else 1
            elif relative<.35:chosen=0 if dark==sn.WALL_Z else 1
            else:confident=False
        if confident:result[m]=candidates[chosen]
        records.append(dict(region=int(k),pixels=int(m.sum()),median_art_intensity=mean,anchor_z=means[sn.WALL_Z],anchor_x=means[sn.WALL_X],shadingRatioZX=None if means[sn.WALL_X] in (None,0) or means[sn.WALL_Z] is None else means[sn.WALL_Z]/means[sn.WALL_X],confident=bool(confident),before=prior.tolist(),after=sn.normalise(result[m].mean(0)).tolist(),family='z-up' if chosen==0 else 'x-up',corrected=bool(confident and chosen!=prior_family)))
    return result,records


def conservative_material(rgb,alpha,baseline,paint,emission,light):
    """Start at today's exact normal/spec/AO; edit only independent feature masks."""
    normal,spec,ao=[x.copy() for x in baseline];valid=(alpha>.5)&~emission
    prob=np.zeros((*alpha.shape,3),np.float32)
    labels=np.stack([normal[...,1],normal[...,2],normal[...,0]],-1).argmax(-1)
    np.put_along_axis(prob,labels[...,None],1,axis=-1)
    detected,ds,da,info=sn.enhanced_building_material(rgb,alpha,prob,paint|emission)
    curves=info.pop('curve_families')
    changed=(curves>0)&valid
    normal[changed]=detected[changed];spec[(curves==1)&valid]=sn.SPEC_GLASS
    slopes,records=paint_planes(paint,valid)
    # A geometrically detected plate is a slope even if a model proposed a wall.
    normal[slopes>0]=sn.normalise(np.array([0.,.60,.80]))
    normal,orientation=orient_slopes(normal,rgb,valid,slopes,light,baseline[0])
    changed|=slopes>0
    # Preserve the existing AO outside confident geometric edits; no broad re-filtering.
    ao[changed]=np.minimum(ao[changed],da[changed])
    family=lambda n:np.stack([n[...,1],n[...,2],n[...,0]],-1).argmax(-1)
    outside=valid&~changed
    report=dict(curves=info['curves'],sloped_regions=records,orientation=orientation,
                changed_pixels_pct=float(100*changed.sum()/max(1,valid.sum())),
                outside_feature_face_agreement_pct=float(100*(family(normal)[outside]==family(baseline[0])[outside]).mean()) if outside.any() else None,
                outside_feature_material_exact=bool(np.array_equal(normal[outside],baseline[0][outside]) and np.array_equal(spec[outside],baseline[1][outside]) and np.array_equal(ao[outside],baseline[2][outside])))
    return normal,spec,ao,curves,slopes,changed,report


def rankdata(a):
    a=np.asarray(a);order=np.argsort(a);ranks=np.empty(len(a),float)
    start=0
    while start<len(a):
        end=start+1
        while end<len(a) and abs(a[order[end]]-a[order[start]])<1e-6:end+=1
        ranks[order[start:end]]=(start+end-1)/2;start=end
    return ranks


def shading_metrics(normal,rgb,valid,light,plane_ids=None,minimum=8):
    """Region Spearman and pixel share passing cross-family shading-order votes.

    Each connected region is weighted equally in Spearman. For the second metric,
    each region votes against all regions with a different orientation family and
    |dot difference| >= .08. A region is consistent only if >=60% comparisons have
    the same brightness order; ties in luminance do not count as correct. The
    denominator includes all opaque pixels, including unmeasured tiny regions.
    """
    if plane_ids is None:plane_ids=plane_regions(normal,valid,minimum)
    lum=rgb@np.array([.2126,.7152,.0722]);regions=[]
    for k in np.unique(plane_ids[valid]):
        if k==0:continue
        count,cc,stats,_=cv2.connectedComponentsWithStats(((plane_ids==k)&valid).astype(np.uint8),8)
        for j in range(1,count):
            m=cc==j
            if m.sum()<minimum:continue
            n=sn.normalise(normal[m].mean(0));family=int(np.argmax([n[1],n[2],n[0]]))
            regions.append(dict(id=int(k),component=j,pixels=int(m.sum()),normal=n.tolist(),dot=float(n@light),luminance=float(lum[m].mean()),family=family))
    dots=np.array([r['dot'] for r in regions]);lums=np.array([r['luminance'] for r in regions])
    dr,lr=rankdata(dots),rankdata(lums)
    rho=float(np.corrcoef(dr,lr)[0,1]) if len(regions)>2 and np.std(dr)>0 and np.std(lr)>0 else None
    share=0;tested=0
    for r in regions:
        comparisons=[q for q in regions if q['family']!=r['family'] and abs(q['dot']-r['dot'])>=.08]
        vote=sum((q['dot']-r['dot'])*(q['luminance']-r['luminance'])>0 for q in comparisons)/max(1,len(comparisons))
        r.update(order_agreement=vote,comparisons=len(comparisons),consistent=bool(comparisons and vote>=.60))
        if comparisons:tested+=r['pixels']
        if r['consistent']:share+=r['pixels']
    return dict(spearman=rho,consistent_opaque_pixels_pct=100*share/max(1,valid.sum()),measured_opaque_pixels_pct=100*sum(r['pixels'] for r in regions)/max(1,valid.sum()),tested_opaque_pixels_pct=100*tested/max(1,valid.sum()),region_count=len(regions),regions=regions)
