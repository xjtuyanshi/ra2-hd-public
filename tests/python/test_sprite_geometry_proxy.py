"""Synthetic projection, material and assist-family checks. No game assets."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[2] / 'packages/ra2-assets/python'))
import sprite_geometry_proxy as gp
import sprite_normals as sn


def test_cell_camera_projects_to_original_diamond():
    c={'origin':[0,0],'pixels_per_world':60/np.sqrt(2)}
    result=gp.project(np.eye(3),c)
    assert np.allclose(result[0],[30,15])
    assert np.allclose(result[2],[-30,15])
    assert np.allclose(result[1],[0,-np.cos(np.pi/6)*60/np.sqrt(2)])


def test_proxy_buffers_have_unit_normals_height_and_empty_background():
    c={'size':[80,80],'origin':[40,40],'pixels_per_world':30,'primitives':[{'kind':'box','center':[0,.5,0],'size':[.8,1,.8]}]}
    b=gp.rasterize(c);m=b['mask']
    assert m.any() and not m.all()
    assert np.allclose(np.linalg.norm(b['normal'][m],axis=-1),1)
    assert b['height'][m].min()>=-1e-6 and b['height'][m].max()<=1.00001
    assert b['ao'][~m].max()==0


def test_snapping_and_existing_assist_rejections_are_independent():
    raw=sn.normalise(np.broadcast_to(np.array([.3,.8,.5]),(8,8,3))).copy()
    families=np.zeros((8,8),np.uint8);families[4:]=2
    snapped,curved=sn.snap_normal_families(raw,families)
    assert (snapped[:4, :, 0]==0).all()  # hard planes stay in a roof/wall plane
    assert (snapped[4:, :, 0]==0).all()  # x-axis cylinders have no axial normal
    assert curved[4:].all() and not curved[:4].any()
    ok,reasons=sn.assist_passes({'iou':.8,'shadingRatioZX':.7,'sidePrior':.4},.88)
    assert not ok and len(reasons)==3


def test_round_art_component_gets_glass_sphere_without_asset_coordinates():
    yy, xx = np.indices((100, 100)); disk = (xx-50)**2+(yy-45)**2<20**2
    rgb = np.full((100,100,3),.25,np.float32); rgb[disk]=[.55,.6,.9]
    alpha = np.ones((100,100),np.float32)
    prob = np.zeros((100,100,3),np.float32); prob[...,1]=1
    normal,spec,ao,info=sn.enhanced_building_material(rgb,alpha,prob)
    assert any(c['kind']=='sphere' for c in info['curves'])
    assert spec[45,50] == 1
    assert np.isfinite(normal).all() and np.isfinite(ao).all()

def test_exact_silhouette_matches_material_rasterizer():
    c={'size':[80,80],'origin':[40,40],'pixels_per_world':30,'primitives':[
        {'kind':'wedge','center':[0,.5,0],'size':[.8,1,.8],'slope_axis':2,'slope':-.5},
        {'kind':'cylinder','center':[.6,.6,.1],'size':[.6,.2,.2],'axis':0},
        {'kind':'sphere','center':[0,1.1,0],'size':[.5,.5,.5]}]}
    assert np.array_equal(gp.silhouette(c),gp.rasterize(c)['mask'])


def test_region_shading_measure_separates_reversed_geometry():
    import sprite_material_regions as mr
    light=np.array([0.,.6,.8]);n=np.zeros((24,36,3));rgb=np.zeros((24,36,3));valid=np.ones((24,36),bool)
    for i,(normal,brightness) in enumerate([([1,0,0],.2),([0,1,0],.6),([0,0,1],.8)]):
        n[:,i*12:(i+1)*12]=normal;rgb[:,i*12:(i+1)*12]=brightness
    good=mr.shading_metrics(n,rgb,valid,light)
    bad=mr.shading_metrics(n[:,::-1],rgb,valid,light)
    assert good['spearman']>.99 and bad['spearman']<-.99
    assert good['consistent_opaque_pixels_pct']==100 and bad['consistent_opaque_pixels_pct']==0


def test_region_shading_tiebreak_fixes_slope_family_not_tilt():
    import sprite_material_regions as mr
    normal=np.zeros((24,36,3));normal[:,:12]=[0,0,1];normal[:,12:24]=[1,0,0];normal[:,24:]=[.8,.6,0]
    rgb=np.ones_like(normal)*.75;rgb[:,:12]=.65;rgb[:,12:24]=.25
    ids=np.zeros((24,36),int);ids[:,24:]=1
    out,records=mr.orient_slopes(normal,rgb,np.ones((24,36),bool),ids,np.array([0,.6,.8]))
    assert np.allclose(out[:,24:],[0,.6,.8]) and records[0]['corrected']
    assert np.array_equal(out[:,:24],normal[:,:24])


def test_conservative_material_keeps_current_buffers_outside_features():
    import sprite_material_regions as mr
    rgb=np.broadcast_to(np.array([.36,.21,.08]),(32,32,3)).copy()
    alpha=np.ones((32,32));normal=np.broadcast_to(np.array([0.,0.,1.]),(32,32,3)).copy()
    spec=np.full((32,32),.08);ao=np.full((32,32),.83);empty=np.zeros((32,32),bool)
    out=mr.conservative_material(rgb,alpha,(normal,spec,ao),empty,empty,np.array([0.,.6,.8]))
    assert not out[5].any()
    assert np.array_equal(out[0],normal) and np.array_equal(out[1],spec) and np.array_equal(out[2],ao)
    assert out[6]['outside_feature_face_agreement_pct']==100


if __name__=='__main__':
    suite=unittest.TestSuite(unittest.FunctionTestCase(f) for n,f in list(globals().items()) if n.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
