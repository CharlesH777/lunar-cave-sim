"""Portable V11 terrain recipe, independent of previous output packages."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def make_scene_config(seed=20261110):
    from build_lunar_cave_v4_cpu import scene_config as base_scene_config
    from cave.lunar_ground_v7 import ground_config
    from cave.lunar_branch_v10 import branch_config
    from cave.lunar_junction_v11 import junction_config
    config = base_scene_config();ground = ground_config()
    config.update(schema_version='lunar_rounded_fork_terrain/11.0',
        candidate_id='rounded_equal_caliber_fork_v11',random_seed=int(seed),
        ground_profile_v7=ground,obstacle_revision=7,branch_v10=branch_config(),
        junction_v11=junction_config(),boulder_detail_revision=10)
    for name in ['cavern','pit']:
        config[name]['ground_profile_v7']=copy.deepcopy(ground)
    config['branch_v10']['ground_profile_v7']=copy.deepcopy(ground)
    config['branch_v10']['junction_floor_deposits']=copy.deepcopy(config['cavern']['floor_deposits'])
    config['vehicle_geometry']=dict(wheel_radius_m=.18,wheelbase_m=.52,track_m=.66,
        overall_length_m=.95,overall_width_m=.85,belly_clearance_m=.15,
        scope='authored visual rover dimensions; geometry screening, not tyre dynamics')
    config['task_spawn']=dict(location='pit_bottom',xy_m=[-15.,-3.],surface_descent_included=False)
    config['task_goal']=dict(xy_m=[74.,-2.6])
    config['task_branch_goal']=dict(xy_m=list(config['branch_v10']['branch_goal_xy_m']))
    config['gravel_collision_v7']=dict(min_full_diameter_m=.25)
    config['obstacle_map_validation']=dict(grid_bounds_xy_m=[[-35.,-23.],[79.,61.]])
    config['runtime_scope']='rounded natural fork terrain with static obstacles; no policy or wheel-dynamics result'
    config['authored_assumptions'].extend([
        'Shared uneven floor with local deposits, with no flattened or pre-cleared driving strip',
        'Two comparable-width underground arms within the original longitudinal extent',
        'Locally rounded CSG residual-rock nose; fillet dimensions are engineering parameters, not lunar measurements',
        'Deterministic rubble layout may require explicitly recorded relocation of a small number of whole objects'])
    entries=[]
    def add(x,y,scale,region='main',group='v10_large_collapse'):
        entries.append(dict(xy_m=[float(x),float(y)],scale_xyz_m=list(scale),region=region,
            group=group,seed=int(seed)+101003*(len(entries)+1)))
    for x,y,scale in [(14.,3.2,(1.35,1.0,.92)),(19.7,2.5,(1.15,.85,.85)),
        (36.,-3.3,(1.25,.92,1.02)),(42.5,-2.5,(1.45,1.0,.95)),
        (61.1,2.9,(1.2,.95,.86)),(69.6,-2.2,(1.2,.90,.88))]:
        add(x,y,scale);add(x+1.6,y+.7,(.65,.51,.46),group='v10_medium_collapse')
    add(-7.,-5.,(1.4,1.05,1.),group='v10_pit_boulder')
    add(-4.8,-4.2,(.75,.60,.54),group='v10_pit_boulder')
    for x,y,scale in [(46.,23.,(1.05,.80,.80)),(55.,36.,(.95,.78,.78)),(63.,39.,(1.10,.82,.85))]:
        add(x,y,scale,region='branch',group='v10_branch_obstacle')
        add(x+1.2,y+.8,(.58,.46,.42),region='branch',group='v10_branch_rubble')
    config['additional_boulders_v10']=entries
    assert config['additional_boulders_v10'][10]['xy_m']==[69.6,-2.2]
    assert 'rover_route' not in json.dumps(config)
    return config


def make_surface_config(seed=20261110):
    return dict(gravel=dict(count=18000,prototype_count=12,seed=int(seed),size_fractions=[.94,.05,.01]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'configs/lunar_fork_v11')
    parser.add_argument('--seed',type=int,default=20261110)
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=False)
    for name,value in [('scene.json',make_scene_config(args.seed)),('surface.json',make_surface_config(args.seed))]:
        path=args.output/name;path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(path)


if __name__=='__main__':main()
