"""Deterministic aperiodic 3D rock roughness, authored in metres.

Hash-grid value noise has no tiled texture or plane-wave construction. The
6/2/.7m octaves use .8/.35/.12m amplitudes. These are simulation parameters,
not measured lunar rock statistics. ``rock_noise`` already returns metres.
"""
from __future__ import annotations
import numpy as np

_MASK=np.uint64(0xffffffff)


def _hash(ix,iy,iz,seed):
    # Explicit 32-bit wrap makes negative cells and scalar/array calls agree.
    a=np.asarray(ix,dtype=np.int64).astype(np.uint64)&_MASK
    b=np.asarray(iy,dtype=np.int64).astype(np.uint64)&_MASK
    c=np.asarray(iz,dtype=np.int64).astype(np.uint64)&_MASK
    with np.errstate(over='ignore'):
        h=((a*np.uint64(0x8da6b343))^(b*np.uint64(0xd8163841))
           ^(c*np.uint64(0xcb1ab31f))^np.uint64(int(seed)&0xffffffff))&_MASK
        h=((h^(h>>np.uint64(16)))*np.uint64(0x7feb352d))&_MASK
        h=((h^(h>>np.uint64(15)))*np.uint64(0x846ca68b))&_MASK
        h=(h^(h>>np.uint64(16)))&_MASK
    return h.astype(np.float64)*(2./4294967295.)-1.


def value_noise(x,y,z,seed=20261110,smooth=True):
    """Broadcast hash-grid noise in [-1,1], quintic smooth or linear faceted."""
    x,y,z=np.asarray(x,dtype=float),np.asarray(y,dtype=float),np.asarray(z,dtype=float)
    ix,iy,iz=np.floor(x).astype(np.int64),np.floor(y).astype(np.int64),np.floor(z).astype(np.int64)
    tx,ty,tz=x-ix,y-iy,z-iz
    if smooth:
        tx=tx**3*(tx*(tx*6.-15.)+10.)
        ty=ty**3*(ty*(ty*6.-15.)+10.)
        tz=tz**3*(tz*(tz*6.-15.)+10.)
    c00=_hash(ix,iy,iz,seed)*(1.-tx)+_hash(ix+1,iy,iz,seed)*tx
    c10=_hash(ix,iy+1,iz,seed)*(1.-tx)+_hash(ix+1,iy+1,iz,seed)*tx
    c01=_hash(ix,iy,iz+1,seed)*(1.-tx)+_hash(ix+1,iy,iz+1,seed)*tx
    c11=_hash(ix,iy+1,iz+1,seed)*(1.-tx)+_hash(ix+1,iy+1,iz+1,seed)*tx
    return (c00*(1.-ty)+c10*ty)*(1.-tz)+(c01*(1.-ty)+c11*ty)*tz


def rock_noise(x,y,z,seed=20261110):
    """Return metre-valued 6/2/.7m fractal roughness; scalar or broadcasting.

    Do not multiply by the legacy .68 gain. Reduced surface/floor amplitudes
    should be explicit at the call site. Fixed orthonormal rotation removes
    a privileged global grid orientation. Absolute-value ridges and linear
    fine interpolation add irregular creases, rather than smooth sine bands.
    """
    x,y,z=np.asarray(x,dtype=float),np.asarray(y,dtype=float),np.asarray(z,dtype=float)
    qx=.8*x+.36*y+.48*z
    qy=-.6*x+.48*y+.64*z
    qz=-.8*y+.6*z
    coarse=value_noise(qx/6.+13.71,qy/6.-7.24,qz/6.+3.13,seed)
    medium=value_noise(qx/2.-3.9,qy/2.+16.8,qz/2.-10.3,seed+104729)
    ridge=np.clip(2.*np.abs(medium)-.5,-1.,1.)
    fine=value_noise(qx/.7+1.2,qy/.7-8.4,qz/.7+24.3,seed+224737,smooth=False)
    return .8*coarse+.35*(.65*medium+.35*ridge)+.12*fine
