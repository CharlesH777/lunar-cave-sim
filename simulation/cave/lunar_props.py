"""Original lunar mission target props and passive exploration instruments.

These assets are separate from the terrestrial cave models. Lunar-specific
hardware is illustrative visual modeling, not flight-qualified equipment.
Coordinates: metres, Z up; target origins sit at the local floor elevation.
"""
from __future__ import annotations

import math
import numpy as np
from pxr import Gf, Sdf, UsdGeom
from .props import (
    _aid, _backpack, _toolbox, _drill, _palette, _mesh, _label,
    material, xform, box, cylinder, ring, tube,
)


def _lunar_palette(stage):
    m = _palette(stage)
    m.update({
        'gold': material(stage,'lunar_MLI_gold',(.64,.36,.075),.3,.78),
        'silver_foil': material(stage,'lunar_silver_foil',(.57,.61,.64),.36,.90),
        'ceramic': material(stage,'lunar_white_ceramic',(.70,.72,.69),.48,.05),
        'solar': material(stage,'lunar_solar_cells',(.009,.025,.069),.21,.45),
        'solar_grid': material(stage,'lunar_cell_grid',(.14,.19,.27),.36,.65),
        'regolith': material(stage,'lunar_sample_rock',(.16,.16,.15),.92),
    })
    return m


def _sample_canister(stage,p,m):
    """Hermetically sealed sample container, 0.32m diameter and 0.39m tall."""
    cylinder(stage,p+'/MainVessel',(0,0,.185),.145,.29,m['silver'],collision=True)
    cylinder(stage,p+'/LowerFlange',(0,0,.047),.162,.026,m['steel'])
    cylinder(stage,p+'/UpperFlange',(0,0,.329),.162,.027,m['silver'])
    cylinder(stage,p+'/Lid',(0,0,.352),.154,.021,m['ceramic'])
    cylinder(stage,p+'/LidCentre',(0,0,.367),.096,.012,m['silver'])
    ring(stage,p+'/SealingGasket',(0,0,.338),.151,.004,m['rubber'])
    ring(stage,p+'/LowerWeld',(0,0,.074),.145,.003,m['chrome'])
    ring(stage,p+'/UpperWeld',(0,0,.30),.145,.003,m['chrome'])
    # Eight flange bolts and raised rib rings make the pressure-vessel construction legible.
    for i in range(8):
        angle=2*math.pi*i/8
        xx=.142*math.cos(angle); yy=.142*math.sin(angle)
        cylinder(stage,p+f'/LidBolt{i}',(xx,yy,.371),.009,.018,m['steel'])
        cylinder(stage,p+f'/LowerBolt{i}',(xx,yy,.033),.008,.013,m['silver'])
    for i,z in enumerate([.11,.155,.2,.245,.285]):
        ring(stage,p+f'/VesselRib{i}',(0,0,z),.146,.003,m['silver_foil'])
    for i,sign in enumerate([-1,1]):
        y=sign*.154
        box(stage,p+f'/HandleMount{i}',(0,y,.22),(.088,.018,.095),m['steel'],.009)
        tube(stage,p+f'/CarryHandle{i}',[(-.045,y,.24),(-.045,y+sign*.069,.25),
                                       (.045,y+sign*.069,.25),(.045,y,.24)],.013,m['silver'])
    # Vacuum port, protected valve and rectangular identification plate.
    cylinder(stage,p+'/ValveStem',(.045,-.035,.39),.012,.04,m['steel'])
    box(stage,p+'/ValveLever',(.045,-.035,.414),(.067,.013,.012),m['blue'],.004)
    cylinder(stage,p+'/VacuumPort',(-.045,.035,.393),.018,.04,m['silver'])
    cylinder(stage,p+'/VacuumCap',(-.045,.035,.418),.023,.014,m['rubber'])
    box(stage,p+'/IDPlate',(-.147,0,.215),(.008,.12,.065),m['ceramic'],.004)
    for i,width in enumerate([.085,.066,.079]):
        box(stage,p+f'/IDLine{i}',(-.152,0,.235-i*.019),(.003,width,.004),m['steel'],.001)
    box(stage,p+'/CautionBand',(-.148,0,.11),(.009,.12,.025),m['yellow'],.004)


def _cylinder_between(stage,path,a,b,r,mat):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    d=b-a;length=float(np.linalg.norm(d))
    tilt=math.degrees(math.acos(float(d[2]/length)))
    yaw=math.degrees(math.atan2(float(d[1]),float(d[0])))
    return cylinder(stage,path,(a+b)/2,r,length,mat,rot=(0,tilt,yaw))


def _communications_tripod(stage,p,m):
    xform(stage,p,(5.,-2.15,.035))
    for i in range(3):
        a=2*math.pi*i/3+.35
        foot=np.array([.42*math.cos(a),.42*math.sin(a),.035])
        top=np.array([.09*math.cos(a),.09*math.sin(a),.8])
        _cylinder_between(stage,p+f'/Leg{i}',foot,top,.019,m['silver'])
        box(stage,p+f'/Foot{i}',foot,(.13,.11,.035),m['rubber'],.008)
        _cylinder_between(stage,p+f'/Brace{i}',foot*.5+[0,0,.38],(0,0,.45),.009,m['steel'])
    cylinder(stage,p+'/Hub',(0,0,.79),.105,.075,m['steel'])
    cylinder(stage,p+'/Mast',(0,0,1.),.026,.39,m['silver'])
    box(stage,p+'/Avionics',(0,.08,.67),(.28,.18,.22),m['gold'],.025)
    box(stage,p+'/AvionicsEnd',(0,.177,.67),(.23,.015,.17),m['silver_foil'],.009)
    for i in range(5):
        box(stage,p+f'/RadiatorFin{i}',(-.10+i*.05,-.026,.67),(.017,.019,.16),m['steel'],.003)
    # Shallow parabolic dish, tilted to face the cave entrance (-X).
    dish=p+'/Dish';xform(stage,dish,(0,0,1.18),(0,-68,0))
    segments,rings=64,12;verts=[(0,0,0)]
    radius=.31;depth=.075
    for j in range(1,rings+1):
        rr=radius*j/rings
        for k in range(segments):
            a=2*math.pi*k/segments
            verts.append((rr*math.cos(a),rr*math.sin(a),depth*(rr/radius)**2))
    faces=[]
    for k in range(segments):faces.append((0,k+1,(k+1)%segments+1))
    for j in range(rings-1):
        lo=1+j*segments;hi=lo+segments
        for k in range(segments):
            kn=(k+1)%segments
            faces.extend([(lo+k,hi+k,hi+kn),(lo+k,hi+kn,lo+kn)])
    obj=_mesh(stage,dish+'/Reflector',verts,faces,m['silver_foil'])
    obj.CreateDoubleSidedAttr(True)
    ring(stage,dish+'/Rim',(0,0,depth),radius,.009,m['silver'])
    for i in range(3):
        a=2*math.pi*i/3
        _cylinder_between(stage,dish+f'/FeedStrut{i}',(.27*math.cos(a),.27*math.sin(a),.058),(0,0,.21),.006,m['steel'])
    cylinder(stage,dish+'/FeedHorn',(0,0,.214),.025,.043,m['ceramic'])
    tube(stage,p+'/DataCable',[(0,.10,.57),(.05,.15,.3),(.05,.2,.04),(.65,.3,.035)],.011,m['rubber'])


def _passive_solar_panel(stage,p,m):
    # A stowed/deployed panel near the entrance, with no artificial emission.
    xform(stage,p,(4.2,-2.05,.035),(0,-12,-8))
    box(stage,p+'/Frame',(0,0,.11),(.66,.84,.045),m['silver'],.015)
    box(stage,p+'/CellBed',(0,0,.137),(.615,.795,.012),m['rubber'],.008)
    for i in range(3):
        for j in range(5):
            xx=-.20+i*.20;yy=-.306+j*.153
            box(stage,p+f'/Cell{i}_{j}',(xx,yy,.145),(.187,.140,.004),m['solar'],.003)
            for k in range(3):
                box(stage,p+f'/Busbar{i}_{j}_{k}',(xx-.06+k*.06,yy,.1475),(.0018,.13,.001),m['solar_grid'],.0002)
    for i,sign in enumerate([-1,1]):
        box(stage,p+f'/Stand{i}',(0,sign*.31,.042),(.49,.025,.05),m['steel'],.005)
    box(stage,p+'/Connector',(.338,.16,.11),(.035,.08,.05),m['rubber'],.007)
    tube(stage,p+'/Cable',[(.35,.16,.11),(.45,.21,.07),(.57,.35,.025),(.68,.43,.025)],.01,m['rubber'])


def build_lunar_props(stage,parent='/World/Targets',positions=None):
    """Build lunar targets; optional id->xy or id->xyz overrides support layouts."""
    positions = positions or {}
    UsdGeom.Xform.Define(stage,parent)
    m=_lunar_palette(stage)
    definitions=[
        ('first_aid_kit','红色应急救援箱',(28.5,-8.5,.05),.34,_aid,(0.,0.,-12.)),
        ('backpack','橙色月面装备包',(16.,-8.5,.05),.29,_backpack,(0.,0.,18.)),
        ('sample_canister','银色密封采样罐',(22.,8.,.05),.25,_sample_canister,(0.,0.,0.)),
        ('toolbox','黄色月面工具箱',(29.,5.,.05),.36,_toolbox,(0.,0.,-14.)),
        ('drill','蓝色取芯电钻',(7.,1.7,.05),.30,_drill,(0.,0.,-25.)),
    ]
    result=[]
    for category,label,pos,radius,fn,rotation in definitions:
        if category in positions:
            override=np.asarray(positions[category],dtype=float)
            if override.shape not in [(2,),(3,)] or not np.isfinite(override).all():
                raise ValueError(f'Invalid lunar target position for {category}')
            pos=tuple(override) if len(override)==3 else (float(override[0]),float(override[1]),.05)
        path=parent+'/'+category
        root=xform(stage,path,pos,rotation)
        _label(root.GetPrim(),category,category)
        root.GetPrim().CreateAttribute('cave:mission_context',Sdf.ValueTypeNames.String).Set('lunar lava tube exploration')
        fn(stage,path,m)
        # Mission patches on textile and rugged transport equipment.
        if category=='backpack':
            box(stage,path+'/MissionPatch',(-.23,0,.49),(.009,.11,.045),m['ceramic'],.004)
            box(stage,path+'/MissionPatchStripe',(-.236,0,.49),(.003,.07,.008),m['blue'],.001)
        if category=='drill':
            cylinder(stage,path+'/CoreBarrel',(-.32,0,.33),.017,.13,m['silver'],'X')
            ring(stage,path+'/CoreBarrelCollar',(-.375,0,.33),.019,.003,m['steel'],'X')
        result.append({'id':category,'prim_path':path,'category':category,'label_zh':label,
                       'position':list(pos),'radius':radius})
    _communications_tripod(stage,'/World/LunarArtifacts/CommunicationsTripod',m)
    _passive_solar_panel(stage,'/World/LunarArtifacts/SolarPanel',m)
    return result
