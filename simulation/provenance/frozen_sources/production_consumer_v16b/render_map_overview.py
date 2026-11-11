"""Render the actual eight cave free-space fields as a standalone map atlas."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cave.layouts import MAPS
from cave.lunar_geometry import configure_layout,get_layout,cave_field,floor_height

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=ROOT/'outputs/maps_revised');args=p.parse_args()
out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
fig,axs=plt.subplots(2,4,figsize=(20,11),facecolor='#0b1118',constrained_layout=True)
x=np.arange(-1,34,.12);y=np.arange(-14,14,.12);xx,yy=np.meshgrid(x,y)
for ax,key in zip(axs.ravel(),MAPS):
    configure_layout(key);layout=get_layout()
    field=cave_field(xx,yy,floor_height(xx,yy)+.65)
    ax.set_facecolor('#0b1118')
    ax.contourf(xx,yy,field,levels=[.07,20],colors=['#59616a'])
    ax.contour(xx,yy,field,levels=[.07],colors=['#a2abb4'],linewidths=.65)
    for i,route in enumerate(layout['branches']):
        route=np.asarray(route);ax.plot(route[:,0],route[:,1],color=['#e4b74e','#6eb7d8','#de8769','#a5c58f','#b59bd0','#dd9fb3'][i%6],lw=1,alpha=.65)
        p=route[len(route)//2];ax.text(p[0],p[1]+.5,chr(65+i),fontsize=8,color='#ffefbf',ha='center')
    for k,p in layout['targets'].items():
        ax.scatter(*p,c='#ff6c63' if k=='first_aid_kit' else '#ddd5b5',s=21 if k=='first_aid_kit' else 10,zorder=5)
    ax.scatter(*layout['spawn'][:2],marker='^',s=40,c='#8be9c6',zorder=6)
    ax.set_title(f'{key} / {layout["name"]}\n{len(layout["branches"])} branches  |  {layout["loop_count"]} loops',color='#e5e7eb',fontsize=11,pad=10)
    ax.set_aspect('equal');ax.set_xlim(-1,34);ax.set_ylim(-14,14)
    ax.tick_params(colors='#7c8794',labelsize=7)
    ax.set_xlabel('metres',color='#7c8794',fontsize=8)
    for spine in ax.spines.values():spine.set_color('#273340')
fig.suptitle('SELENE / Eight authored lunar lava-tube training maps\nActual procedural free-space field · green = rover start · red = rescue kit · pale = other search objects',color='#e8e5d9',fontsize=18)
fig.savefig(out/'map_atlas.png',dpi=160,facecolor=fig.get_facecolor())
plt.close(fig)
print(out/'map_atlas.png')
