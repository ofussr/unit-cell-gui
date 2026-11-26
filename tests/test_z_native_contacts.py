"""Analytic circle/convex face clipping, independent of OpenGL."""
import numpy as np
import pytest
from unit_cell_gui._contacts import ContactFace, native_positions, CIRCLE_COS, CIRCLE_SIN
from unit_cell_gui._ucg_native import contact_arcs
from unit_cell_gui._spatial import BoundsIndex


def frame(poly):
    return ContactFace(np.array(poly,float),np.array([0.,0.,1.]),np.array([1.,0.,0.]),np.array([0.,1.,0.]))


def rectangle(x0,x1,y0,y1):
    return np.array([[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]],float)


def run(f, centre=(0,0,0), radius=1., signed=0.):
    return native_positions(f,np.array([centre]),np.array([radius]),np.array([signed]),contact_arcs)


def test_full_circle_preserves_grid_and_order():
    f=frame(rectangle(-2,2,-2,2));p=run(f)
    grid=np.column_stack((CIRCLE_COS,CIRCLE_SIN,np.zeros(160)))
    np.testing.assert_array_equal(p,np.stack((grid,np.roll(grid,-1,axis=0)),axis=1).reshape(-1,3))


def test_half_circle_exact_boundaries_both_windings():
    poly=rectangle(0,2,-2,2)
    for points in (poly,poly[::-1]):
        p=run(frame(points))
        assert len(p)==160
        assert np.min(p[:,0])>=-1e-14
        np.testing.assert_allclose(np.linalg.norm(p,axis=1),1.,atol=1e-14)
        for y in (-1,1):assert np.any(np.linalg.norm(p-[0,y,0],axis=1)<1e-12)


def test_short_arc_between_old_samples_and_tangent():
    angle=.009; c,s=np.cos(angle),np.sin(angle)
    rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    poly=rectangle(np.cos(.002),1.1,-.1,.1)@rot.T
    f=frame(poly);p=run(f)
    assert p.shape==(2,3)
    np.testing.assert_allclose(np.linalg.norm(p,axis=1),1.,atol=1e-14)
    np.testing.assert_allclose((p@rot)[:,0],np.cos(.002),atol=1e-13)
    grid=np.column_stack((CIRCLE_COS,CIRCLE_SIN,np.zeros(160)))
    assert not np.any(f.contains((grid+np.roll(grid,-1,axis=0))/2))
    assert not len(run(frame(rectangle(1,2,-1,1))))
    assert not len(run(frame(rectangle(2,3,-1,1))))


def test_random_rectangles_against_dense_angular_reference():
    rng=np.random.default_rng(746)
    t=np.arange(36000)*(2*np.pi/36000)
    unit=np.column_stack((np.cos(t),np.sin(t),np.zeros(len(t))))
    for _ in range(100):
        lo=rng.uniform(-1.5,.4,2);hi=lo+rng.uniform(.1,2.,2)
        poly=rectangle(lo[0],hi[0],lo[1],hi[1]);f=frame(poly)
        p=run(f);ends=p.reshape(-1,2,3)
        expected=np.mean(np.all((unit[:,:2]>=lo)&(unit[:,:2]<=hi),axis=1))*2*np.pi
        actual=np.sum(np.arctan2(np.linalg.norm(np.cross(ends[:,0],ends[:,1]),axis=1),np.sum(ends[:,0]*ends[:,1],axis=1)))
        assert abs(actual-expected)<.001
        assert np.all(p[:,:2]>=lo-1e-12) and np.all(p[:,:2]<=hi+1e-12)
        np.testing.assert_allclose(np.linalg.norm(p,axis=1),1.,atol=1e-12)


def test_translated_plane_and_nonzero_signed_distance():
    shift=np.array([1e6,-1e6,4e5]);f=frame(rectangle(0,2,-2,2)+shift)
    p=run(f,shift+[0,0,.6],1.,.6)-shift
    np.testing.assert_allclose(p[:,2],0,atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(p,axis=1),.8,atol=2e-10)
    assert p[:,0].min()>=-1e-10


def test_bvh_batch_refits_match_fresh_index_and_exhaustive():
    rng=np.random.default_rng(824);low=rng.uniform(-10,10,(173,3));high=low+1
    index=BoundsIndex(low,high)
    for _ in range(12):
        ids=rng.choice(len(low),size=53,replace=False);low[ids]=rng.uniform(-20,20,(len(ids),3));high[ids]=low[ids]+rng.uniform(0,3,(len(ids),3));low[ids[:3]]=np.inf;high[ids[:3]]=-np.inf
        index.update(ids,low[ids],high[ids]);fresh=BoundsIndex(low,high)
        for _ in range(12):
            a=rng.uniform(-15,15,3);b=a+8
            expected=np.flatnonzero(np.all(low<=high,axis=1)&np.all(low<=b,axis=1)&np.all(high>=a,axis=1))
            np.testing.assert_array_equal(index.query(a,b),expected)
            np.testing.assert_array_equal(index.query(a,b),fresh.query(a,b))
