"""Render scientific geometry QA plots on CPU; these are not Isaac images."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from cave.network3d import evaluate_field


COLORS = {"main": "#438eae", "rejoining_branch": "#dc9a38", "lower_level_branch": "#a87bd4"}


def overview(output: Path):
    network = json.loads((output/"network.json").read_text())
    report = json.loads((output/"geometry_report.json").read_text())
    data = np.load(output/"mesh.npz")
    vertices, faces = data["vertices"], data["faces"]
    # Cluster only for a transparent overview. The exported collision mesh is untouched.
    cells = np.rint(vertices/2.5).astype(np.int32)
    _, labels = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(labels)
    clustered = np.column_stack([np.bincount(labels, weights=vertices[:,i])/counts for i in range(3)])
    mapped = labels[faces]
    good = (mapped[:,0] != mapped[:,1]) & (mapped[:,1] != mapped[:,2]) & (mapped[:,0] != mapped[:,2])
    surface = np.unique(np.sort(mapped[good], axis=1), axis=0)

    fig = plt.figure(figsize=(18,11), facecolor="#f7f8fa")
    gs = fig.add_gridspec(2,3, height_ratios=(1.1,1), hspace=.32, wspace=.27)
    ax3 = fig.add_subplot(gs[0,:2], projection="3d")
    axplan = fig.add_subplot(gs[0,2])
    axside = fig.add_subplot(gs[1,:2])
    axcross = fig.add_subplot(gs[1,2])
    shell = Poly3DCollection(clustered[surface], facecolors="#8499a9", edgecolors="none", alpha=.52,
                             rasterized=True)
    ax3.add_collection3d(shell)
    used = set()
    for edge in network["edges"]:
        p = np.asarray(edge["floor_centerline_xyz_m"], dtype=float)
        role = edge.get("role", "main")
        color = COLORS.get(role, "#667788")
        label = role.replace("_", " ") if role not in used else None
        used.add(role)
        ax3.plot(*p.T, color=color, lw=3, label=label)
        axplan.plot(*p[:,:2].T, color=color, lw=2.4)
        axside.plot(p[:,0], p[:,2], color=color, lw=2.6, marker=".", ms=4)
    for node in network["nodes"]:
        x,y,z = node["xyz_m"]
        ax3.text(x,y,z+3,node["id"], fontsize=9)
        axplan.annotate(node["id"], (x,y), xytext=(4,5), textcoords="offset points", fontsize=9)
    lo,hi = vertices.min(0), vertices.max(0)
    ax3.set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), zlim=(lo[2]-2,hi[2]+4),
            xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    ax3.set_box_aspect(np.maximum(hi-lo,1), zoom=1.28)
    ax3.set_zticks([float(lo[2]), float(hi[2])], labels=[f"{lo[2]:.0f}",f"{hi[2]:.0f}"])
    ax3.set_yticks([0.] if lo[1] <= 0. <= hi[1] else [float((lo[1]+hi[1])/2)])
    ax3.tick_params(axis="both", labelsize=8, pad=1)
    ax3.set_xlabel("X (m)", labelpad=10)
    ax3.view_init(elev=25, azim=-65)
    ax3.set_title("3D cavity mesh + floor reference lines (equal metric scales)", fontsize=12)
    ax3.legend(loc="upper left", fontsize=9)
    axplan.set(title="Plan: projected paths", xlabel="X (m)", ylabel="Y (m)")
    axplan.set_aspect("equal", adjustable="datalim")
    axplan.grid(alpha=.2)

    for crossing in network.get("nonconnecting_crossings", []):
        x,y = crossing["xy_m"]
        axplan.scatter([x],[y], marker="x", s=70, color="#c34646", zorder=10)
        axplan.annotate("crossing, no connection",(x,y), xytext=(-130,-25), textcoords="offset points",
                        fontsize=8, arrowprops={"arrowstyle":"->", "color":"#c34646"})
        zz = np.linspace(lo[2]-2, hi[2]+2, 2500)
        air = evaluate_field(network, np.column_stack([np.full(len(zz),x),np.full(len(zz),y),zz])) > 0
        transitions = np.diff(np.r_[False,air,False].astype(np.int8))
        starts, ends = np.flatnonzero(transitions==1), np.flatnonzero(transitions==-1)-1
        intervals = [(float(zz[a]),float(zz[b])) for a,b in zip(starts,ends)]
        for z0,z1 in intervals:
            axside.plot([x,x],[z0,z1], color="#c34646", lw=5, alpha=.7)
        if len(intervals)==2:
            gap = intervals[1][0]-intervals[0][1]
            for measured in report.get("nonconnecting_crossings", []):
                if measured["edge_a"] == crossing["edge_a"] and measured["edge_b"] == crossing["edge_b"]:
                    gap = measured["minimum_actual_rock_thickness_m"]
            axside.annotate(f"Separate cavities\nrock gap ~{gap:.1f} m", (x,(intervals[0][1]+intervals[1][0])/2),
                            xytext=(x+25, intervals[0][1]+5), fontsize=10,
                            arrowprops={"arrowstyle":"->", "color":"#c34646"})
    side_title = "Side projection (expanded vertical scale): floor elevations"
    if network.get("nonconnecting_crossings"):
        side_title += "; red bars show air at crossing"
    axside.set(xlabel="X (m)", ylabel="Z (m)", title=side_title)
    axside.grid(alpha=.2)

    # Plane through a main-trunk segment, away from junctions, sampled from the same actual 3D field.
    a,b = np.array([35.,-4.,-147.]), np.array([65.,3.,-146.])
    midpoint=(a+b)/2
    tangent=(b-a)/np.linalg.norm(b-a)
    lateral=np.array([-tangent[1],tangent[0],0.]);lateral/=np.linalg.norm(lateral)
    up=np.cross(tangent,lateral);up/=np.linalg.norm(up)
    uu=np.linspace(-19,21,241); vv=np.linspace(-2,19,169)
    u,v=np.meshgrid(uu,vv)
    pts=midpoint+u[...,None]*lateral+v[...,None]*up
    field=evaluate_field(network,pts.reshape(-1,3)).reshape(u.shape)
    axcross.contourf(u,v,field>0,levels=[-.5,.5,1.5], colors=["#dfe2e6", "#fcfcfc"])
    axcross.contour(u,v,field,levels=[0], colors=["#384f60"], linewidths=1.6)
    axcross.plot([0],[0],marker="+",ms=9,color="#c34646")
    axcross.set(xlabel="Lateral offset (m)", ylabel="Height above floor reference (m)",
                title="Actual implicit cross-section\nasymmetric sides + roof indentation")
    axcross.set_aspect("equal")
    axcross.grid(alpha=.15)
    fig.suptitle(f"{network['candidate_id']}: {network['title']}", fontsize=18, fontweight="bold", y=.98)
    fig.text(.5,.018,"SYNTHETIC HYPOTHESIS  |  Lunar section scale + terrestrial morphology priors  |  CPU geometry view, not an Isaac render or navigation result",
             ha="center",fontsize=10,color="#4e5965")
    fig.savefig(output/"overview.png",dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    overview(args.output)
    print(args.output/"overview.png")


if __name__=="__main__":
    main()
