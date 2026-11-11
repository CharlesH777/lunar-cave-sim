"""Chinese overhead map from saved V10 terrain, boulders and checked paths.

This is a map of actual scene geometry. Paths are geometric search results,
not cleared lanes, policy rollouts or added road markings inside the cave.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _xy(value):
    if isinstance(value, dict):
        value=value.get('declared_xy_m',value.get('xy_m',value.get('xyz_m')))
    result=np.asarray(value,float)
    if result.ndim!=1 or len(result)<2 or not np.isfinite(result[:2]).all():
        raise ValueError('Invalid declared map point')
    return result[:2]


def _path(report):
    if report.get('passed') is not True or report.get('path_found') is not True:
        return None
    result=np.asarray(report.get('path_xyz_m'),float)
    if result.ndim!=2 or result.shape[1] not in (2,3) or len(result)<2 or not np.isfinite(result).all():
        raise ValueError('A passing path report is missing finite actual path points')
    return result[:,:2]


def _place_labels(ax, labels, font):
    """Choose deterministic label offsets using actual renderer text bounds."""
    from matplotlib.transforms import Bbox
    figure=ax.figure;figure.canvas.draw();renderer=figure.canvas.get_renderer()
    scale=figure.dpi/72.;bounds=ax.get_window_extent(renderer);occupied=[]
    for item in labels:
        text=item['text'];xy=np.asarray(item['xy'],float);color=item['color']
        width,height,_=renderer.get_text_width_height_descent(text,font,False)
        width+=8*scale;height+=7*scale
        width_points=width/scale
        candidates=[(11,17),(11,37),(-width_points-11,17),(-width_points-11,37),
                    (11,-31),(-width_points-11,-31),(21,60),(-width_points-21,60),
                    (21,-54),(-width_points-21,-54),(-width_points*.5,73)]
        anchor=ax.transData.transform(xy);best=None
        for dx,dy in candidates:
            box=Bbox.from_bounds(anchor[0]+dx*scale-4*scale,anchor[1]+dy*scale-3*scale,width,height)
            overlap=0.
            for other in occupied:
                intersection=Bbox.intersection(box,other)
                if intersection is not None:overlap+=intersection.width*intersection.height
            clipped=Bbox.intersection(box,bounds)
            outside=box.width*box.height-(0. if clipped is None else clipped.width*clipped.height)
            score=overlap+outside*4.+.035*(dx*dx+dy*dy)
            if best is None or score<best[0]:best=(score,dx,dy,box)
        _,dx,dy,box=best;occupied.append(box)
        ax.annotate(text,xy,xytext=(dx,dy),textcoords='offset points',ha='left',va='bottom',
            fontproperties=font,color='#162432',zorder=12,
            bbox=dict(boxstyle='round,pad=.28',fc='white',ec=color,lw=1.15,alpha=.94),
            arrowprops=dict(arrowstyle='-',color=color,lw=1.15,shrinkA=3,shrinkB=5))


def plot_map(source,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager,patheffects
    from matplotlib.patches import Circle,Patch
    from matplotlib.lines import Line2D

    source=Path(source).expanduser().resolve(strict=True)
    output=Path(output).expanduser().absolute()
    font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not font_path.is_file():font_path=Path('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    if not font_path.is_file():raise FileNotFoundError('A Chinese Noto/WenQuanYi font is required')
    font=font_manager.FontProperties(fname=str(font_path),size=12)
    small=font_manager.FontProperties(fname=str(font_path),size=10)
    title_font=font_manager.FontProperties(fname=str(font_path),size=19)
    plt.rcParams['axes.unicode_minus']=False
    files={name:source/name for name in ['targets.json','boulders.json','scene_config.json',
        'validation/grid_data.npz','validation/obstacle_map_report.json',
        'validation_branch/grid_data.npz','validation_branch/obstacle_map_report.json']}
    hashes={name:_sha(path) for name,path in files.items()}
    targets=_read(files['targets.json'])
    targets=targets['targets'] if isinstance(targets,dict) else targets
    boulders=_read(files['boulders.json']);config=_read(files['scene_config.json'])
    main=_read(files['validation/obstacle_map_report.json'])
    branch=_read(files['validation_branch/obstacle_map_report.json'])
    with np.load(files['validation/grid_data.npz']) as data:
        x,y,height,terrain=[np.asarray(data[k]) for k in ['x','y','height','terrain_free']]
        slope=np.asarray(data['slope_deg']) if 'slope_deg' in data else None
    with np.load(files['validation_branch/grid_data.npz']) as data:
        if not np.array_equal(x,data['x']) or not np.array_equal(y,data['y']):
            raise ValueError('Main and branch maps must share the actual scene grid')
    if height.shape!=(len(x),len(y)) or terrain.shape!=height.shape:
        raise ValueError('Invalid terrain grid shapes')
    visible_floor=np.isfinite(height)&(height<3.)
    # Clip only the display colour range. Actual saved heights remain unchanged;
    # steep wall shoulders stay grey rather than flattening them into a roadway.
    if slope is not None:visible_floor&=np.isfinite(slope)&(slope<35.)
    values=height[visible_floor]
    if not len(values):raise ValueError('No finite underground floor heights to draw')
    lo,hi=np.percentile(values,[1.,99.])
    if hi-lo<.12:lo-=.06;hi+=.06
    step_x=float(x[1]-x[0]);step_y=float(y[1]-y[0])
    extent=[float(x[0]-step_x*.5),float(x[-1]+step_x*.5),float(y[0]-step_y*.5),float(y[-1]+step_y*.5)]
    figure,ax=plt.subplots(figsize=(17,10),facecolor='#f4f6f8')
    figure.subplots_adjust(left=.065,right=.89,bottom=.14,top=.84)
    ax.set_facecolor('#b6bdc5')
    image=ax.imshow(np.ma.masked_where(~visible_floor,height).T,origin='lower',extent=extent,
        cmap='cividis',vmin=float(lo),vmax=float(hi),interpolation='nearest',aspect='equal',zorder=1)
    limited=visible_floor&~terrain
    ax.imshow(np.where(limited.T,1.,np.nan),origin='lower',extent=extent,cmap='Greys',
        vmin=0,vmax=1,alpha=.27,interpolation='nearest',zorder=2)
    gravel_path=source/'obstacle_report.json';gravel_count=0
    if gravel_path.is_file():
        for item in _read(gravel_path).get('obstacles',[]):
            center=np.asarray(item['center_xyz_m'],float)
            if center[2]>=22.:continue
            ax.add_patch(Circle(center[:2],float(item['xy_radius_m']),fc='#61656b',ec='none',alpha=.67,zorder=3))
            gravel_count+=1
    floor_boulder_count=0;new_boulder_count=0
    for item in boulders:
        center=np.asarray(item['center_xyz_m'],float);scale=np.asarray(item['scale_xyz_m'],float)
        if center.shape!=(3,) or scale.shape!=(3,) or not np.isfinite(np.r_[center,scale]).all():
            raise ValueError('Malformed boulder map record')
        if center[2]>=22.:continue
        radius=1.35*float(np.linalg.norm(scale[:2]))
        new=str(item.get('group','')).startswith('v10_')
        ax.add_patch(Circle(center[:2],radius,fc='#775136' if new else '#3d4249',
            ec='#f3c975' if new else '#b7c0ca',lw=1.15 if new else .55,alpha=.92,zorder=4))
        floor_boulder_count+=1;new_boulder_count+=int(new)
    path_rows=[]
    for report,color,dashed,label in [(main,'#05e5ea',False,'通道A几何路线'),(branch,'#ff9d32',True,'通道B几何路线')]:
        path=_path(report)
        if path is None:continue
        line,=ax.plot(path[:,0],path[:,1],color=color,lw=3.1 if not dashed else 2.6,
            linestyle=(0,(5,2)) if dashed else '-',solid_capstyle='round',zorder=6)
        line.set_path_effects([patheffects.Stroke(linewidth=4.7,foreground='#202c39',alpha=.70),patheffects.Normal()])
        path_rows.append(dict(label=label,length_m=report.get('path_length_m'),point_count=len(path)))
    colors={'astronaut':'#ee8740','lunar_rover':'#2fbf66','lunar_base':'#a175e0','chinese_flag':'#e84145'}
    labels=[]
    for item in targets:
        xy=_xy(item.get('xyz_m',item.get('position')));radius=float(item.get('footprint_radius_m',item.get('local_ground_footprint_radius_m',.5)))
        category=item.get('class_name',item.get('id',''))
        color=colors.get(category,'#ce4ad0')
        ax.add_patch(Circle(xy,radius,fc=color,ec='white',lw=1.3,alpha=.90,zorder=7))
        ax.scatter(*xy,s=38,c=color,edgecolors='white',linewidths=1.,zorder=8)
        labels.append(dict(xy=xy,text=item.get('label_zh',category),color=color))
        approach=item.get('approach_xy_m')
        if approach is not None:
            approach=_xy(approach);ax.plot([xy[0],approach[0]],[xy[1],approach[1]],color=color,lw=1.2,ls=':',zorder=7)
            ax.scatter(*approach,c=color,marker='x',s=35,zorder=8)
    start=_xy(main['start']);main_goal=_xy(main['goal']);branch_goal=_xy(branch['goal'])
    for xy,marker,color,label in [(start,'*','#f9fafb','坑底起点'),(main_goal,'X','#05e5ea','通道A终点'),(branch_goal,'X','#ff9d32','通道B终点')]:
        ax.scatter(*xy,marker=marker,s=165 if marker=='*' else 85,c=color,edgecolors='#202b36',linewidths=1.,zorder=10)
        labels.append(dict(xy=xy,text=label,color='#344b60' if marker=='*' else color))
    ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.grid(alpha=.17,color='white',lw=.65)
    ax.set_xlabel('局部世界坐标 X（米）',fontproperties=font)
    ax.set_ylabel('局部世界坐标 Y（米）',fontproperties=font)
    ax.tick_params(labelsize=11)
    title='月洞俯视布局｜原80米纵向范围保留·自然双通道'
    if main.get('passed') is not True or branch.get('passed') is not True:title+='（路线验证未全部通过）'
    figure.suptitle(title,fontproperties=title_font,x=.49,y=.967,color='#1d2d3d')
    seed=config.get('search_layout_seed',config.get('random_seed','—'))
    figure.text(.50,.923,f'实际地形高程 · 大块落石与碎石 · 随机物品布局（种子 {seed}）',
        fontproperties=font,ha='center',color='#455666')
    legend=[Line2D([0],[0],color='#05e5ea',lw=3,label='通道A几何路线'),
        Line2D([0],[0],color='#ff9d32',lw=3,ls='--',label='通道B几何路线'),
        Patch(fc='#b6bdc5',label='洞壁／坡度与车辆净空限制'),
        Patch(fc='#3d4249',ec='#b7c0ca',label='原大石保守占地'),
        Patch(fc='#775136',ec='#f3c975',label='新增大块落石'),
        Line2D([0],[0],color='#72767c',marker='o',lw=0,ms=4,label='中大碎石')]
    ax.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,1.015),ncol=3,
        prop=small,frameon=False,columnspacing=1.6)
    colorbar=figure.colorbar(image,ax=ax,shrink=.64,pad=.022,fraction=.028)
    colorbar.set_label('真实地面高程（米；颜色范围按1–99分位截取）',fontproperties=small,labelpad=12)
    colorbar.ax.tick_params(labelsize=10)
    _place_labels(ax,labels,font)
    lengths=' / '.join(f"{row['label']} {row['length_m']:.1f}米" for row in path_rows if row['length_m'] is not None)
    figure.text(.07,.070,lengths or '尚无通过完整复核的路线',fontproperties=font,color='#31465a')
    figure.text(.07,.040,'圆圈是物体的保守占地，叉号是可接近点；路线为实际网格几何搜索，并非预先清平车道或模型导航成绩。',
        fontproperties=small,color='#4b5e70')
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output,dpi=145,facecolor=figure.get_facecolor());plt.close(figure)
    if any(_sha(path)!=hashes[name] for name,path in files.items()):raise RuntimeError('Map plotting altered a source asset')
    report=dict(passed=True,source=str(source),output=str(output),updated_utc=datetime.now(timezone.utc).isoformat(),
        pixel_size=[2465,1450],source_hashes=hashes,source_files_unchanged=True,
        actual_terrain_heights_used=True,authored_flat_lane=False,paths=path_rows,
        underground_boulder_count=floor_boulder_count,new_v10_boulder_count=new_boulder_count,
        mapped_gravel_count=gravel_count,target_count=len(targets),Chinese_font=str(font_path),GPU_work=False)
    output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(plot_map(args.source,args.output),ensure_ascii=False))


if __name__=='__main__':main()
