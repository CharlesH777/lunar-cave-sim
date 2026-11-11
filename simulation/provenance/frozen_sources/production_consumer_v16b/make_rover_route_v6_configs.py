"""Create an independent pit-bottom driving-route hypothesis for the small rover."""
import argparse
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'configs/rover_route_v6')
    args=parser.parse_args()
    from build_lunar_cave_v4_cpu import scene_config
    from cave.rover_route_v6 import route_config,route_y,route_z
    config=scene_config();route=route_config()
    config.update(schema_version='lunar_rover_route/6.0',candidate_id='pit_bottom_drive_01',
                  rover_route_revision=6,rover_route=route)
    config['cavern']['rover_route_floor']=copy.deepcopy(route)
    config['pit']['rover_route_floor']=copy.deepcopy(route)
    config['authored_assumptions'].append(
        'An authored low-gradient floor corridor for a small rover; this is an engineering affordance, not an observed lunar roadway.')
    config['vehicle_geometry']=dict(wheel_radius_m=.18,wheelbase_m=.52,track_m=.66,
        overall_length_m=.95,overall_width_m=.85,belly_clearance_m=.15,
        scope='dimensions of the locally authored visual rover; no suspension or tyre model')
    first,last=map(float,route['drive_x_range_m'])
    config['task_spawn']=dict(location='pit_bottom',xyz_m=[first,float(route_y(route,first)),float(route_z(route,first))],
        z_reference='ground reference, not camera or rover centre',surface_descent_included=False)
    config['task_goal']=dict(xyz_m=[last,float(route_y(route,last)),float(route_z(route,last))],
                             z_reference='ground reference')
    config['runtime_scope']='pit-bottom route geometry screening; not wheel dynamics or learned navigation evaluation'
    surface=dict(gravel=dict(count=15000,prototype_count=12,seed=20261110))
    args.output.mkdir(parents=True,exist_ok=False)
    for name,value in [('scene.json',config),('surface.json',surface)]:
        path=args.output/name;path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(path)


if __name__=='__main__':main()
