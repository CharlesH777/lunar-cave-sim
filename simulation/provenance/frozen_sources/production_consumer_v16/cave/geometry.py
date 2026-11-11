"""Deterministic, connected limestone cave. No simulator imports; metres/Z-up."""
import numpy as np
from skimage.measure import marching_cubes


def floor_height(x, y):
    return .025*np.sin(np.asarray(x)*.74)*np.cos(np.asarray(y)*1.2) + .012*np.sin(np.asarray(y)*3.1+np.asarray(x))


def cave_field(x, y, z):
    """Positive in air. Same continuous geometry drives mesh and clearance."""
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    center = .42*np.sin(x*.29)
    width = 2.85 + .60*np.sin(x*.38+.5) + .35*np.cos(x*.9)
    height = 2.65 + .45*np.sin(x*.24)
    field = (1-np.sqrt(((y-center)/width)**2+((z-2.25)/height)**2))*2.7
    for cx, cy, cz, rx, ry, rz in [(11,-3.8,2.25,4.3,3.5,3.1),(21,3.3,2.5,5,3.4,3.5),(27,-1,2.4,3.8,3.6,3.)]:
        room = (1-np.sqrt(((x-cx)/rx)**2+((y-cy)/ry)**2+((z-cz)/rz)**2))*2.8
        # Smooth maximum, eliminating artificial intersections between rooms.
        k=.55
        h=np.clip(.5+.5*(field-room)/k,0,1)
        field=room*(1-h)+field*h+k*h*(1-h)
    # Multiscale continuous weathering, strata and fissures.
    n=.17*np.sin(x*1.5+np.sin(y*1.1))*np.sin(z*1.9+y*.5)
    n+=.09*np.sin(x*4.2+y*2.2+np.sin(z*3))*np.sin(z*4.5-y)
    n+=.035*np.sin(x*11.4+y*5.7)*np.sin(z*10.3+x*.8)
    n+=.075*np.sin(z*9.4+.6*np.sin(x*.8)+y*.2)
    field=field+n
    return np.minimum.reduce(np.broadcast_arrays(field,z-floor_height(x,y),x+1.2,31.3-x))


def inside_cave(x,y,z):
    return cave_field(x,y,z)>.06


def build_shell(spacing=.13):
    origin=np.array([-1.7,-8.5,-.35])
    xx=np.arange(origin[0],31.85,spacing,dtype=np.float32)
    yy=np.arange(origin[1],8.5,spacing,dtype=np.float32)
    zz=np.arange(origin[2],7.4,spacing,dtype=np.float32)
    field=cave_field(xx[:,None,None],yy[None,:,None],zz[None,None,:]).astype('float32')
    v,f,n,_=marching_cubes(field,0,spacing=(spacing,)*3,gradient_direction='descent')
    v+=origin
    cent=v[f].mean(1)
    is_floor=cent[:,2] < floor_height(cent[:,0],cent[:,1])+.10
    return v,f,is_floor


def ceiling_height(x,y):
    z=np.arange(1,7,.035)
    ids=np.flatnonzero(cave_field(x,y,z)>0)
    return float(z[ids[-1]]) if len(ids) else 2.0


def formation_mesh(x,y,z,length,radius,down,rng):
    rings, sides=22,14
    t=np.linspace(0,1,rings)
    angle=np.arange(sides)*2*np.pi/sides
    taper=(1-t)**1.7
    # Irregular calcite drips instead of perfect cones.
    rr=radius*taper[:,None]*(1+.12*np.sin(angle[None,:]*5+t[:,None]*13))+.006
    bend=.06*np.sin(t*2.4)*length
    vertices=np.stack(np.broadcast_arrays(x+rr*np.cos(angle)+bend[:,None],
                         y+rr*np.sin(angle)+.04*np.sin(t[:,None]*5),
                         z+(-1 if down else 1)*length*t[:,None]),axis=-1).reshape(-1,3)
    faces=[]
    for j in range(rings-1):
        for k in range(sides):
            a=j*sides+k;b=j*sides+(k+1)%sides;c=b+sides;d=a+sides
            faces.extend([(a,b,c),(a,c,d)])
    faces=np.array(faces,dtype=np.int32)
    if down:faces=faces[:,::-1].copy()
    return vertices,faces


def concatenate(parts):
    vs,fs=[],[]; off=0
    for v,f in parts:
        vs.append(v);fs.append(f+off);off+=len(v)
    return np.concatenate(vs),np.concatenate(fs)
