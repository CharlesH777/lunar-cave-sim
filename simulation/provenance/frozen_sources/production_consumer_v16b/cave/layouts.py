"""Eight authored lava-tube graphs, rather than seed variants of one mesh.

All maps share the entrance and main gallery. Their branch connections, loops,
dead ends and object search destinations differ. Coordinates are metres/Z-up.
"""
from copy import deepcopy
import math


def _layout(name, branches, destinations, loops):
    junctions=[]
    for i,route in enumerate(branches):
        code=chr(65+i)
        junctions.append({"position":list(route[0]),"destination":list(route[1]),
                          "label":f"ROUTE {code} / {'NORTH' if route[1][1]>0 else 'SOUTH'} GALLERY",
                          "branch_id":code})
        if abs(route[-1][1])<.01:
            junctions.append({"position":list(route[-1]),"destination":list(route[-2]),
                              "label":f"ROUTE {code} / LOOP RETURN","branch_id":code})
    keys=["first_aid_kit","backpack","sample_canister","toolbox","drill"]
    return {"name":name,"branches":[[list(p) for p in route] for route in branches],
            "targets":dict(zip(keys,[list(p) for p in destinations])),
            "spawn":[2.,.3,0.],"main":[[0.,0.],[31.,0.]],
            "junctions":junctions,"loop_count":loops,
            "bounds":[-.5,34.,-14.,14.]}


MAPS={
    "01":_layout("Tranquillity Forks",[
        [(8,0),(11,-6),(17,-9)],
        [(15,0),(17,6),(24,9),(28,5),(26,0)],
        [(23,0),(25,-6),(29,-9)],
    ],[(28.5,-8.5),(16,-8.5),(22,8),(29,5),(7,1.7)],1),
    "02":_layout("Four Blind Galleries",[
        [(6,0),(6,-6),(11,-8)],
        [(12,0),(10,6),(5,8)],
        [(20,0),(22,-5),(29,-7)],
        [(25,0),(25,6),(30,9)],
    ],[(28,-6.7),(10,-7.5),(6,7.6),(29,8.4),(7,1.7)],0),
    "03":_layout("Twin Circuit",[
        [(7,0),(7,7),(14,8),(16,0)],
        [(18,0),(17,-7),(26,-8),(28,0)],
        [(24,0),(25,6),(30,9)],
    ],[(25,-7.8),(11.5,7.6),(28.5,8),(20,-7.3),(7,1.7)],2),
    "04":_layout("Basalt Comb",[
        [(6,0),(6,-7)],[(10,0),(10,7)],
        [(15,0),(15,-7),(18,-10)],
        [(20,0),(20,7),(17,10)],
        [(25,0),(25,-7)],[(29,0),(29,8)],
    ],[(17.5,-9.5),(6,-6.2),(17.5,9.5),(29,7.2),(10,6)],0),
    "05":_layout("Long Perimeter",[
        [(6,0),(5,-6),(11,-11),(23,-11),(29,-6),(28,0)],
        [(13,0),(12,6),(7,10)],
        [(22,0),(23,7),(29,10)],
        [(18,0),(18,-5)],
    ],[(20,-11),(8,9.2),(28,9.5),(18,-4.2),(7,1.7)],1),
    "06":_layout("Offset Rings",[
        [(9,0),(7,5),(11,9),(17,6),(18,0)],
        [(11,0),(10,-6),(16,-10),(23,-6),(24,0)],
        [(26,0),(28,6),(31,9)],
        [(29,0),(29,-7)],
    ],[(16,-10),(11,9),(30,8),(29,-6.2),(7,1.7)],2),
    "07":_layout("Broken Ridge",[
        [(5,0),(4,-5),(8,-9),(13,-10)],
        [(12,0),(12,5),(17,10),(23,9),(22,0)],
        [(20,0),(20,-6),(25,-10),(30,-8)],
        [(28,0),(29,5),(31,8)],
        [(6,0),(5,5),(3,8)],
    ],[(28,-8.8),(11,-9.6),(19,9.6),(30,7),(4.5,5.5)],1),
    "08":_layout("Asymmetric Figure Eight",[
        [(5,0),(4,6),(11,11),(18,8),(20,0)],
        [(8,0),(8,-6),(14,-11),(24,-10),(29,-4),(28,0)],
        [(27,0),(28,5),(31,9)],
        [(19,0),(19,-5)],
    ],[(22,-10.2),(9,9.5),(30,7.5),(19,-4.2),(7,1.7)],2),
}


def _fit_targets_to_narrow_passages(layout):
    """Preserve each search region while fitting original wide-cave placements.

    The road lengths, junctions, loops and robot size are unchanged. Objects sit
    on a passage shoulder rather than being left inside the newly closer walls.
    """
    layout['targets_wide_version']=deepcopy(layout['targets'])
    routes=[layout['main']]+layout['branches']
    for index,(key,point) in enumerate(layout['targets'].items()):
        px,py=point
        best=None
        for route_index,route in enumerate(routes):
            for a,b in zip(route[:-1],route[1:]):
                dx,dy=b[0]-a[0],b[1]-a[1]
                length2=dx*dx+dy*dy
                t=max(0.,min(1.,((px-a[0])*dx+(py-a[1])*dy)/length2))
                qx,qy=a[0]+t*dx,a[1]+t*dy
                distance2=(px-qx)**2+(py-qy)**2
                if best is None or distance2<best[0]:best=(distance2,qx,qy,dx,dy,route_index)
        _,qx,qy,dx,dy,route_index=best
        length=math.hypot(dx,dy);nx,ny=-dy/length,dx/length
        side=1. if (px-qx)*nx+(py-qy)*ny>=0 else -1.
        # Wider main gallery leaves room to pass the drill; branch targets can
        # be approached from the nearest side without occupying the centreline.
        offset=.62 if route_index==0 else .46
        layout['targets'][key]=[round(qx+side*offset*nx,4),round(qy+side*offset*ny,4)]
    layout['geometry_revision']='narrow_passages_v2'
    layout['nominal_width_m']={'main':[2.4,3.0],'branch':[1.8,2.2]}


for _record in MAPS.values():
    _fit_targets_to_narrow_passages(_record)


def get_layout(map_id="01"):
    key=str(map_id).zfill(2)
    if key not in MAPS:
        raise ValueError(f"Unknown lunar map {map_id}; choose 01 through 08")
    result=deepcopy(MAPS[key]);result["id"]=key
    return result
