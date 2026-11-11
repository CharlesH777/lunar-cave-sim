"""Author uneven ground and retained obstacles without reserving a driving lane."""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'configs/lunar_obstacle_v7')
    args = parser.parse_args()
    from build_lunar_cave_v4_cpu import scene_config
    from cave.lunar_ground_v7 import ground_config
    config = scene_config()
    ground = ground_config()
    config.update(schema_version='lunar_obstacle_terrain/7.0', candidate_id='uneven_ground_obstacles_01',
                  ground_profile_v7=ground, obstacle_revision=7)
    for name in ('cavern', 'pit'):
        config[name]['ground_profile_v7'] = copy.deepcopy(ground)
    config['authored_assumptions'].append(
        'Moderate shared two-dimensional floor relief and local deposits; no flattened lane or route-conditioned obstacle removal.')
    config['vehicle_geometry'] = dict(wheel_radius_m=.18, wheelbase_m=.52, track_m=.66,
        overall_length_m=.95, overall_width_m=.85, belly_clearance_m=.15,
        scope='authored visual rover dimensions; no suspension or tyre dynamics')
    config['task_spawn'] = dict(location='pit_bottom', xy_m=[-15., -3.], surface_descent_included=False)
    config['task_goal'] = dict(xy_m=[74., -2.6])
    config['runtime_scope'] = 'uneven terrain with static obstacles; geometric free-path screening only'
    config['gravel_collision_v7'] = dict(min_full_diameter_m=.25)
    assert 'rover_route' not in json.dumps(config)
    options = dict(gravel=dict(count=15000, prototype_count=12, seed=20261110))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('scene.json', config), ('surface.json', options)]:
        path = args.output/name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        print(path)


if __name__ == '__main__':
    main()
