"""Procedural lunar lava-tube analogue, skylight, and overlying regolith.

An authored training environment, not a reconstruction of an observed lunar cave.
Positive field is free space; same geometry is used for mesh and navigation.
"""
import numpy as np
import math
from skimage.measure import marching_cubes
from .layouts import MAPS, get_layout as _get_layout


# Centre-line coordinates are shared with route maps/sign placement. A and C
# terminate underground; B reconnects to the main gallery and forms a real loop.
_ACTIVE_MAP_ID="01"
BRANCH_ROUTES={chr(65+i):route for i,route in enumerate(MAPS["01"]["branches"])}


def configure_layout(map_id="01"):
    global _ACTIVE_MAP_ID,BRANCH_ROUTES
    layout=_get_layout(map_id)
    _ACTIVE_MAP_ID=layout["id"]
    BRANCH_ROUTES={chr(65+i):route for i,route in enumerate(layout["branches"])}
    return layout


def get_layout(map_id=None):
    return _get_layout(_ACTIVE_MAP_ID if map_id is None else map_id)


def _smooth_union(a,b,k=.16):
    h=np.clip(.5+.5*(a-b)/k,0,1)
    return b*(1-h)+a*h+k*h*(1-h)


def _branch_segment(x,y,z,start,end):
    """Capped elliptical passage; all dimensions are metres.

    The lower section has about 2.0–2.2 m clear width at rover body height.
    Rounded caps produce closed dead ends rather than openings in the boundary.
    """
    ax,ay=start;bx,by=end
    dx,dy=bx-ax,by-ay
    t=np.clip(((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy),0.,1.)
    distance=np.sqrt((x-(ax+t*dx))**2+(y-(ay+t*dy))**2)
    radius=1.14+.04*np.sin((ax+t*dx)*.8+(ay+t*dy)*.4)
    return (1-np.sqrt((distance/radius)**2+((z-.95)/1.95)**2))*2.8


def main_center_y(x):
    """Mild lateral variation; layouts retain their nominal straight spine."""
    return .12*math.sin(float(x)*.22) if np.isscalar(x) else .12*np.sin(np.asarray(x)*.22)


def _scalar_field(x,y,z):
    """Same field using scalar math, for fast cached collision queries."""
    sin,cos,sqrt=math.sin,math.cos,math.sqrt
    def union(a,b,k=.16):
        h=max(0.,min(1.,.5+.5*(a-b)/k))
        return b*(1-h)+a*h+k*h*(1-h)
    center=.12*sin(x*.22)
    width=1.45+.10*sin(x*.34)+.04*cos(x*.85)
    high=2.1+.08*sin(x*.21)
    field=(1-sqrt(((y-center)/width)**2+((z-1.05)/high)**2))*2.8
    for route in BRANCH_ROUTES.values():
        for (ax,ay),(bx,by) in zip(route[:-1],route[1:]):
            dx,dy=bx-ax,by-ay
            t=max(0.,min(1.,((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy)))
            distance=sqrt((x-ax-t*dx)**2+(y-ay-t*dy)**2)
            radius=1.14+.04*sin((ax+t*dx)*.8+(ay+t*dy)*.4)
            segment=(1-sqrt((distance/radius)**2+((z-.95)/1.95)**2))*2.8
            field=union(field,segment)
    field+=.13*sin(x*1.7+sin(y*1.6))*sin(z*2.4+y*.8)
    field+=.065*sin(x*5.2+y*2.3)*sin(z*5.8+x*.5)
    field+=.035*sin(x*12.4+y*7.1)*sin(z*11.3)
    field+=.045*sin(z*7.3+x*.11+y*.2)
    floor=.035*sin(x*.75)*cos(y*.8)+.018*sin(x*3.3+y*.7)
    field=min(field,z-floor,x+1.2,31.3-x)
    shaft=min(2.05-sqrt((x-3.2)**2+(y-.15)**2),z-floor)
    surface=7.65+.20*sin(x*.22)*cos(y*.25)+.075*sin(x*.75+y*.6)
    for cx,cy,r in [(-17,5,7),(28,-23,9),(42,25,11)]:
        d=sqrt((x-cx)**2+(y-cy)**2)
        surface-=1.4*math.exp(-(d/(r*.76))**4)
        surface+=.40*math.exp(-((d-r)/(r*.16))**2)
    return max(field,shaft,z-surface)


def floor_height(x,y):
    if np.isscalar(x) and np.isscalar(y):
        return .035*math.sin(x*.75)*math.cos(y*.8)+.018*math.sin(x*3.3+y*.7)
    x,y=np.asarray(x),np.asarray(y)
    return .035*np.sin(x*.75)*np.cos(y*.8)+.018*np.sin(x*3.3+y*.7)


def surface_height(x,y):
    x,y=np.asarray(x),np.asarray(y)
    h=7.65+.20*np.sin(x*.22)*np.cos(y*.25)+.075*np.sin(x*.75+y*.6)
    for cx,cy,r in [(-17,5,7),(28,-23,9),(42,25,11)]:
        d=np.sqrt((x-cx)**2+(y-cy)**2)
        h-=1.4*np.exp(-(d/(r*.76))**4)
        h+=.40*np.exp(-((d-r)/(r*.16))**2)
    return h


def cave_field(x,y,z):
    if np.ndim(x)==0 and np.ndim(y)==0 and np.ndim(z)==0:
        return _scalar_field(float(x),float(y),float(z))
    x,y,z=np.asarray(x),np.asarray(y),np.asarray(z)
    center=.12*np.sin(x*.22)
    width=1.45+.10*np.sin(x*.34)+.04*np.cos(x*.85)
    high=2.1+.08*np.sin(x*.21)
    field=(1-np.sqrt(((y-center)/width)**2+((z-1.05)/high)**2))*2.8
    for route in BRANCH_ROUTES.values():
        for start,end in zip(route[:-1],route[1:]):
            field=_smooth_union(field,_branch_segment(x,y,z,start,end))
    # Broad cooled-flow scallops and angular fractures; no calcite speleothems.
    field+=.13*np.sin(x*1.7+np.sin(y*1.6))*np.sin(z*2.4+y*.8)
    field+=.065*np.sin(x*5.2+y*2.3)*np.sin(z*5.8+x*.5)
    field+=.035*np.sin(x*12.4+y*7.1)*np.sin(z*11.3)
    field+=.045*np.sin(z*7.3+x*.11+y*.2)
    # Clip the whole union after adding branches, so every gallery shares the
    # ground-height contract used by the rover and search object placement.
    field=np.minimum.reduce(np.broadcast_arrays(field,z-floor_height(x,y),x+1.2,31.3-x))
    shaft=2.05-np.sqrt((x-3.2)**2+(y-.15)**2)
    shaft=np.minimum(shaft,z-floor_height(x,y))
    return np.maximum.reduce(np.broadcast_arrays(field,shaft,z-surface_height(x,y)))


def build_shell(spacing=.12):
    origin=np.array([-5.,-15.,-.4])
    xx=np.arange(origin[0],36,spacing,dtype=np.float32)
    yy=np.arange(origin[1],15,spacing,dtype=np.float32)
    zz=np.arange(origin[2],10.2,spacing,dtype=np.float32)
    field=cave_field(xx[:,None,None],yy[None,:,None],zz[None,None,:]).astype('float32')
    v,f,_,_=marching_cubes(field,0,spacing=(spacing,)*3,gradient_direction='descent')
    v+=origin;c=v[f].mean(1)
    regolith=(c[:,2]<floor_height(c[:,0],c[:,1])+.09)|(c[:,2]>surface_height(c[:,0],c[:,1])-.09)
    return v,f,regolith
