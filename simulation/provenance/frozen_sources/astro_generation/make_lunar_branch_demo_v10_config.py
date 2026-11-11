"""Keep the 80m main cavity, add one lateral branch and natural large obstacles."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'configs/lunar_branch_v10')
    args=parser.parse_args()
    from cave.lunar_branch_v10 import branch_config
    config=json.loads((ROOT/'configs/lunar_obstacle_v7/scene.json').read_text())
    config.update(schema_version='lunar_branch_obstacle_demo/10.0',candidate_id='natural_equal_width_fork_02',
        branch_v10=branch_config(),boulder_detail_revision=10)
    config['task_branch_goal']={'xy_m':config['branch_v10']['branch_goal_xy_m']}
    config['obstacle_map_validation']={'grid_bounds_xy_m':[[-35.,-23.],[79.,61.]]}
    config['authored_assumptions'].append('Two comparable-width irregular passages and natural collapse obstacles; original 80m longitudinal extent is retained.')
    entries=[]
    def add(x,y,scale,region='main',group='v10_large_collapse'):
        entries.append(dict(xy_m=[float(x),float(y)],scale_xyz_m=list(scale),region=region,group=group,
            seed=20261110+101003*(len(entries)+1)))
    # Staggered piles are geological obstacles, with no lane carved through them.
    for x,y,scale in [(14.,3.2,(1.35,1.0,.92)),(19.7,2.5,(1.15,.85,.85)),
        (36.,-3.3,(1.25,.92,1.02)),(42.5,-2.5,(1.45,1.0,.95)),
        (61.1,2.9,(1.2,.95,.86)),(69.6,-2.2,(1.2,.90,.88))]:
        add(x,y,scale);add(x+1.6,y+.7,(.65,.51,.46),group='v10_medium_collapse')
    add(-7.,-5.,(1.4,1.05,1.0),group='v10_pit_boulder')
    add(-4.8,-4.2,(.75,.60,.54),group='v10_pit_boulder')
    for x,y,scale in [(46.,23.,(1.05,.80,.80)),(55.,36.,(.95,.78,.78)),
                      (63.,39.,(1.10,.82,.85))]:
        add(x,y,scale,region='branch',group='v10_branch_obstacle')
        add(x+1.2,y+.8,(.58,.46,.42),region='branch',group='v10_branch_rubble')
    config['additional_boulders_v10']=entries
    assert 'rover_route' not in json.dumps(config)
    options={'gravel':{'count':18000,'prototype_count':12,'seed':20261110,
                       'size_fractions':[.94,.05,.01]}}
    args.output.mkdir(parents=True,exist_ok=False)
    for name,value in [('scene.json',config),('surface.json',options)]:
        path=args.output/name;path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(path)


if __name__=='__main__':main()
