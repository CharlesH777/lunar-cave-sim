"""Apply scanned CC0 gravel PBR to a derived lunar stage, without saving it.

The caller owns the stage edit target and final export. Original scan files are
copied byte-for-byte; neutralization and slope blending happen in USD shaders.
RTX evaluation of varying UVTexture scale/bias must be visually accepted by the
caller: the CPU checks here only validate authored connections and arrays.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [('BasaltWalls', '/World/Materials/Basalt', 'wall'),
           ('RegolithFloorAndSurface', '/World/Materials/Regolith', 'floor')]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _linear_scan_mean(path):
    """Exact per-channel sRGB EOTF mean, row-chunked to bound memory."""
    with Image.open(path) as image:
        image = image.convert('RGB')
        width, height = image.size
        total = np.zeros(3, dtype=np.float64)
        for row in range(0, height, 128):
            rgb = np.asarray(image.crop((0,row,width,min(row+128,height))), np.float32)/255.
            linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
            total += linear.sum(axis=(0,1), dtype=np.float64)
    return total/(width*height), [width,height]


@lru_cache(maxsize=4)
def _cached_scan_stats(path):
    mean,resolution=_linear_scan_mean(path)
    return tuple(map(float,mean)),tuple(resolution)


def _choose_resolution(requested=None):
    requested='auto' if requested is None else str(requested).lower()
    requested={'2048':'2k','4096':'4k'}.get(requested,requested)
    if requested not in ['auto','2k','4k']:
        raise ValueError('texture_resolution must be auto, 2k or 4k')
    complete_4k=all((ROOT/'assets/textures'/f'gravel_floor_04_{suffix}_4k.jpg').is_file()
                    for suffix in ['diff','rough','nor_gl'])
    if requested=='auto':
        return '4k' if complete_4k and (ROOT/'assets/GRAVEL_V5_SOURCES.json').is_file() else '2k'
    if requested=='4k' and not complete_4k:
        raise FileNotFoundError('All three 4k gravel PBR maps must be present')
    return requested


def _source_assets(output,resolution):
    catalog = json.loads((ROOT/'assets/ASSET_SOURCES.json').read_text())
    records = {a['id']:a for a in catalog['assets']}
    extra=(json.loads((ROOT/'assets/GRAVEL_V5_SOURCES.json').read_text()) if resolution=='4k' else None)
    dest = output/'textures'
    dest.mkdir(parents=True, exist_ok=True)
    result = {}
    for asset_name, suffixes in [('gravel_floor_04',['diff','rough','nor_gl']),
                                 ('dark_rock',['nor_gl'])]:
        asset = records[asset_name]
        for suffix in suffixes:
            image_resolution=resolution if asset_name=='gravel_floor_04' else '2k'
            name=f'{asset_name}_{suffix}_{image_resolution}.jpg'
            if asset_name=='gravel_floor_04' and resolution=='4k':
                entry=next(f for f in extra['files'] if f['map']==suffix)
                source=ROOT/entry['path']
            else:
                entry=next(f for f in asset['files'] if Path(f['path']).name==name)
                source=ROOT/'assets'/entry['path']
            if not source.is_file() or _sha(source)!=entry['sha256']:
                raise ValueError('Existing scan does not match its source manifest: '+name)
            if entry.get('official_md5') and hashlib.md5(source.read_bytes()).hexdigest()!=entry['official_md5']:
                raise ValueError('4k scan does not match official provider MD5: '+name)
            target=dest/name
            if target.exists():
                if _sha(target)!=entry['sha256']:
                    raise FileExistsError('Refusing to replace a different copied scan: '+str(target))
            else:
                shutil.copy2(source,target)
            result[(asset_name,suffix)]=dict(asset_id=asset_name,map=suffix,
                local_path=str(target.resolve()),source_path=str(source),
                source_uri=asset['source_url'],download_uri=entry['url'],license=asset['license'],
                license_uri=asset['license_canonical_url'],sha256=entry['sha256'],
                bytes=target.stat().st_size,resolution=entry['resolution'],
                official_md5=entry.get('official_md5'),
                provider=asset['provider'],authors=asset.get('authors',[]),original_pixels_unchanged=True)
    return result


def _reader(stage,path,varname,kind,fallback):
    shader=UsdShade.Shader.Define(stage,path)
    suffix={2:'float2',4:'float4'}[kind]
    dtype={2:Sdf.ValueTypeNames.Float2,4:Sdf.ValueTypeNames.Float4}[kind]
    vector={2:Gf.Vec2f,4:Gf.Vec4f}[kind]
    shader.CreateIdAttr('UsdPrimvarReader_'+suffix)
    shader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set(varname)
    shader.CreateInput('fallback',dtype).Set(vector(*fallback))
    shader.CreateOutput('result',dtype)
    return shader


def _texture(stage,path,asset,uv,space):
    shader=UsdShade.Shader.Define(stage,path)
    shader.CreateIdAttr('UsdUVTexture')
    # Absolute copied target paths are deliberate until caller flattens/rebases.
    shader.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(asset['local_path']))
    shader.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set(space)
    shader.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
    shader.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
    shader.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
    shader.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    shader.CreateOutput('r',Sdf.ValueTypeNames.Float)
    return shader


def _normal_coefficients(strength):
    strength=np.asarray(strength,np.float32)
    scale=np.ones((len(strength),4),np.float32)
    bias=np.zeros_like(scale)
    scale[:,:3]=2*strength[:,None]
    bias[:,:2]=-strength[:,None]
    bias[:,2]=1-2*strength
    return scale,bias


def _primvar(mesh,name,values,kind,interpolation='vertex'):
    values=np.asarray(values,np.float32)
    dtype={2:Sdf.ValueTypeNames.TexCoord2fArray,4:Sdf.ValueTypeNames.Float4Array}[kind]
    array={2:Vt.Vec2fArray,4:Vt.Vec4fArray}[kind]
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(name,dtype,interpolation).Set(array.FromNumpy(values))


def _author_mesh_arrays(mesh,cache,gain,tile,wall_tile,floor_strength,wall_strength,low,high):
    points=np.asarray(mesh.GetPointsAttr().Get(),np.float32)
    normals=np.asarray(mesh.GetNormalsAttr().Get(),np.float32)
    if normals.shape!=points.shape or mesh.GetNormalsInterpolation()!='vertex':
        raise ValueError('Terrain must retain original global vertex normals: '+str(mesh.GetPath()))
    matrix=np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()),np.float64)
    world=points.astype(np.float64)@matrix[:3,:3]+matrix[3,:3]
    normal=normals.astype(np.float64)@np.linalg.inv(matrix[:3,:3]).T
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    weight=np.clip((normal[:,2]-low)/(high-low),0.,1.)
    weight=(weight*weight*(3.-2.*weight)).astype(np.float32)
    colors=np.asarray(mesh.GetDisplayColorPrimvar().Get(),np.float32)
    if colors.shape==(1,3):colors=np.broadcast_to(colors,points.shape).copy()
    if colors.shape!=points.shape or not np.isfinite(colors).all():
        raise ValueError('Existing terrain colors must be finite per-vertex displayColor')
    scale=np.ones((len(points),4),np.float32);bias=np.zeros_like(scale)
    scale[:,:3]=weight[:,None]*gain[None,:]
    bias[:,:3]=(1-weight[:,None])*colors
    _primvar(mesh,'ground_st',world[:,:2]/tile,2)
    _primvar(mesh,'gravel_scale',scale,4)
    _primvar(mesh,'gravel_bias',bias,4)
    floor_scale,floor_bias=_normal_coefficients(weight*floor_strength)
    wall_scale,wall_bias=_normal_coefficients((1-weight)*wall_strength)
    for name,value in [('gravel_normal_scale',floor_scale),('gravel_normal_bias',floor_bias),
                        ('rock_normal_scale',wall_scale),('rock_normal_bias',wall_bias)]:
        _primvar(mesh,name,value,4)
    # Separate dominant-plane face UVs give wall normal detail a real 2.4m scale.
    faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),np.int32).reshape(-1,3)
    tri=world[faces]
    fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    dominant=np.abs(fn).argmax(1)
    uv=np.empty((len(faces),3,2),np.float32)
    for axis,coords in [(0,[1,2]),(1,[0,2]),(2,[0,1])]:
        take=dominant==axis
        uv[take]=tri[take][:,:,coords]/wall_tile
    _primvar(mesh,'rock_st',uv.reshape(-1,2),2,'faceVarying')
    return dict(world=np.asarray(world,np.float32),normals=np.asarray(normal,np.float32),
                colors=colors,weight=weight,scale=scale,bias=bias,
                ground_st=np.asarray(world[:,:2]/tile,np.float32))


def set_gravel_primvars(mesh, albedo_gain_rgb=None, tile_m=2.5, *,
                       floor_normal_strength=.9, wall_normal_strength=.3,
                       wall_tile_m=2.4, normal_z_range=(.12,.88), texture_resolution=None):
    """Give derived background meshes the identical world-anchored PBR inputs.

    Pass ``report['gravel_stats']['per_channel_gain']`` from the source to make
    newly created exterior patches use the exact same scan normalization.
    This function does not bind or change any material or save the stage.
    """
    mesh=mesh if isinstance(mesh,UsdGeom.Mesh) else UsdGeom.Mesh(mesh)
    if not mesh:raise ValueError('Expected a valid USD mesh')
    selected_resolution=_choose_resolution(texture_resolution)
    if albedo_gain_rgb is None:
        mean,_=_cached_scan_stats(str(ROOT/'assets/textures'/f'gravel_floor_04_diff_{selected_resolution}.jpg'))
        gain=.135/np.asarray(mean,np.float64)
    else:
        gain=np.asarray(albedo_gain_rgb,np.float64)
    tile=float(tile_m);wall_tile=float(wall_tile_m)
    low,high=map(float,normal_z_range)
    if gain.shape!=(3,) or not np.isfinite(gain).all() or np.any(gain<=0) or tile<=0 or wall_tile<=0 or not low<high:
        raise ValueError('Invalid gravel gain, tile or normal blending range')
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    data=_author_mesh_arrays(mesh,cache,gain.astype(np.float32),tile,wall_tile,
                             float(floor_normal_strength),float(wall_normal_strength),low,high)
    return dict(passed=True,path=str(mesh.GetPath()),vertices=len(data['world']),
                tile_m=tile,albedo_gain_rgb=gain.tolist(),texture_source_resolution=selected_resolution,
                floor_weight_range=[float(data['weight'].min()),float(data['weight'].max())],
                world_xy_anchored=True,material_binding_changed=False,stage_saved=False,
                primvars=['ground_st','gravel_scale','gravel_bias','gravel_normal_scale',
                          'gravel_normal_bias','rock_normal_scale','rock_normal_bias','rock_st'])


def _shared_continuity(arrays):
    if len(arrays)<2:return dict(shared_vertices=0,passed=True)
    a,b=arrays[:2]
    dtype=np.dtype([('x',np.float32),('y',np.float32),('z',np.float32)])
    aa=np.ascontiguousarray(a['world']).view(dtype).reshape(-1)
    bb=np.ascontiguousarray(b['world']).view(dtype).reshape(-1)
    _,ia,ib=np.intersect1d(aa,bb,return_indices=True)
    differences={key:float(np.max(np.abs(a[key][ia]-b[key][ib]))) if len(ia) else 0.
                 for key in ['normals','colors','weight','scale','bias','ground_st']}
    passed=all(v<=2e-6 for v in differences.values())
    if not passed:raise ValueError('Source material partitions disagree at shared vertices: '+repr(differences))
    return dict(shared_vertices=len(ia),passed=passed,maximum_differences=differences)


def apply_gravel_materials(stage, output_dir, config=None):
    """Edit caller-owned derived stage and copy scans; never saves or starts Kit."""
    config={} if config is None else dict(config)
    output=Path(output_dir).expanduser().absolute()
    output.mkdir(parents=True,exist_ok=True)
    tile=float(config.get('gravel_tile_m',2.5))
    wall_tile=float(config.get('wall_micro_tile_m',2.4))
    floor_strength=float(config.get('gravel_normal_strength',.9))
    wall_strength=float(config.get('wall_micro_normal_strength',.3))
    low,high=config.get('gravel_weight_normal_z_range',[.12,.88])
    low,high=float(low),float(high)
    target=np.asarray(config.get('gravel_target_linear_mean',.135),np.float64)
    if target.ndim==0:target=np.full(3,float(target))
    if target.shape!=(3,) or not np.isfinite(target).all() or np.any(target<=0) or tile<=0 or wall_tile<=0 or not low<high:
        raise ValueError('Invalid gravel tile, target mean, or slope blending range')
    if not 0<=floor_strength<=1 or not 0<=wall_strength<=1:
        raise ValueError('Normal strengths must be in [0,1]')
    selected_resolution=_choose_resolution(config.get('texture_resolution'))
    assets=_source_assets(output,selected_resolution)
    mean,resolution=_cached_scan_stats(assets[('gravel_floor_04','diff')]['source_path'])
    mean=np.asarray(mean,np.float64);resolution=list(resolution)
    if np.any(mean<=1e-5):raise ValueError('Gravel diffuse scan has no usable linear RGB mean')
    gain=(target/mean).astype(np.float32)
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    reports=[];arrays=[]
    for name,material_path,role in TARGETS:
        mesh=UsdGeom.Mesh.Get(stage,'/World/Cave/'+name)
        if not mesh:continue
        data=_author_mesh_arrays(mesh,cache,gain,tile,wall_tile,floor_strength,wall_strength,low,high)
        arrays.append(data)
        material=UsdShade.Material.Get(stage,material_path)
        if not material:raise ValueError('Missing original terrain material '+material_path)
        surface=UsdShade.Shader.Get(stage,material_path+'/Surface')
        if not surface:raise ValueError('Missing original surface shader '+material_path)
        ground_uv=_reader(stage,material_path+'/GroundUVV5','ground_st',2,(0.,0.))
        scale=_reader(stage,material_path+'/GravelScaleV5','gravel_scale',4,(0.,0.,0.,1.))
        bias=_reader(stage,material_path+'/GravelBiasV5','gravel_bias',4,(.105,.11,.12,0.))
        diffuse=_texture(stage,material_path+'/GravelDiffuseV5',assets[('gravel_floor_04','diff')],ground_uv,'sRGB')
        diffuse.CreateInput('scale',Sdf.ValueTypeNames.Float4).ConnectToSource(scale.ConnectableAPI(),'result')
        diffuse.CreateInput('bias',Sdf.ValueTypeNames.Float4).ConnectToSource(bias.ConnectableAPI(),'result')
        surface.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuse.ConnectableAPI(),'rgb')
        if role=='floor':
            normal_uv=ground_uv;normal_asset=assets[('gravel_floor_04','nor_gl')];prefix='gravel'
            rough=_texture(stage,material_path+'/GravelRoughV5',assets[('gravel_floor_04','rough')],ground_uv,'raw')
            rough.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.1,.1,.1,1.))
            rough.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.9,.9,.9,0.))
            surface.CreateInput('roughness',Sdf.ValueTypeNames.Float).ConnectToSource(rough.ConnectableAPI(),'r')
        else:
            normal_uv=_reader(stage,material_path+'/RockUVV5','rock_st',2,(0.,0.))
            normal_asset=assets[('dark_rock','nor_gl')];prefix='rock'
            surface.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.95)
        normal=_texture(stage,material_path+'/GroundNormalV5',normal_asset,normal_uv,'raw')
        ns=_reader(stage,material_path+'/NormalScaleV5',prefix+'_normal_scale',4,(0.,0.,0.,1.))
        nb=_reader(stage,material_path+'/NormalBiasV5',prefix+'_normal_bias',4,(0.,0.,1.,0.))
        normal.CreateInput('scale',Sdf.ValueTypeNames.Float4).ConnectToSource(ns.ConnectableAPI(),'result')
        normal.CreateInput('bias',Sdf.ValueTypeNames.Float4).ConnectToSource(nb.ConnectableAPI(),'result')
        surface.CreateInput('normal',Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal.ConnectableAPI(),'rgb')
        connected,name_out,_=surface.GetInput('diffuseColor').GetConnectedSource()
        if str(connected.GetPrim().GetPath())!=str(diffuse.GetPath()) or name_out!='rgb':
            raise ValueError('Scanned diffuse is not connected to terrain surface')
        for field,expected in [('scale',scale),('bias',bias)]:
            connected,name_out,_=diffuse.GetInput(field).GetConnectedSource()
            if str(connected.GetPrim().GetPath())!=str(expected.GetPath()) or name_out!='result':
                raise ValueError('Varying texture '+field+' connection was not authored')
        reports.append(dict(path=str(mesh.GetPath()),material_path=material_path,vertices=len(data['world']),
                            world_xy_gravel_uv=True,weight_range=[float(data['weight'].min()),float(data['weight'].max())],
                            scanned_diffuse_connected=True,varying_scale_bias_connected=True))
    if not reports:raise ValueError('No expected lunar terrain meshes were found')
    continuity=_shared_continuity(arrays)
    return dict(passed=True,material_paths=[r['material_path'] for r in reports],meshes=reports,
        texture_sources=list(assets.values()),gravel_stats=dict(source_srgb_to_linear_mean=mean.tolist(),
            target_linear_mean=target.tolist(),per_channel_gain=gain.tolist(),resolution=resolution,
            texture_resolution=selected_resolution,
            normalization='shader scale only; original scan files unchanged'),
        texture_source_resolution=selected_resolution,
        gravel_tile_m=tile,normal_settings=dict(gravel_strength=floor_strength,wall_strength=wall_strength,
            wall_tile_m=wall_tile,slope_weighted_strength=True,floor_roughness_range=[.9,1.]),
        shared_vertex_continuity=continuity,
        diffuse_formula='linearScanRGB * (floorWeight * gainRGB) + (1-floorWeight) * existingBaseColorRGB',
        authored_texture_paths='absolute copied output textures; caller may rebase after export',
        validation_scope='CPU PXR graph/primvar/source hash checks; caller performs final export and RTX visual acceptance',
        requires_rtx_varying_scale_bias_acceptance=True,
        rtx_fallback='If RTX ignores varying UVTexture scale/bias, replace with an RTX-supported MaterialX mix network; do not silently use unweighted scan or hard material split.',
        original_stage_saved=False,GPU_work=False,simulation_started=False)
