"""Current RGB-D fork proof for routes anywhere inside the wide cave arms."""

import numpy as np

from .lunar_fork_evidence_v12 import ForkEvidence as _DepthWitness, _validated_path


class ForkEvidence(_DepthWitness):
    """Verify the actual guide's arm by its air volume, then current depth.

    A wide arm contains valid routes several metres from its single validation
    centreline. The chosen arm therefore needs an exclusive-air witness, while
    the two path reports still bind the opening geometry and source hashes.
    The inherited frame check retains two visible opening patches, a visible
    separating rock witness, angular separation and camera-relative ordering.
    """

    def __init__(self, world, target_id, guide_xy):
        self.witnesses = []
        self.chosen = None
        self.availability_reason = 'not_initialized'
        try:
            paths = [_validated_path(world, name) for name in ('validation', 'validation_branch')]
            choice = {'passage_A': 0, 'passage_B': 1}.get(world.targets[target_id]['placement_region'])
            guide = np.asarray(guide_xy, dtype=float)
            nose_x = float(world.config['junction_v11']['support_center_xy_m'][0])
            if choice is None or guide.ndim != 2 or guide.shape[1] < 2 or not np.isfinite(guide).all():
                self.availability_reason = 'missing_valid_chosen_arm'
                return
            downstream = guide[guide[:, 0] >= nose_x-12., :2]
            if not len(downstream):
                self.availability_reason = 'route_ends_before_fork'
                return
            from .lunar_ground_v7 import cavern_floor
            from .lunar_morphology_v4 import evaluate_field as main_field
            from .lunar_branch_v10 import evaluate_branch_field
            floor = cavern_floor(world.config['ground_profile_v7'], world.config['cavern'],
                                  downstream[:, 0], downstream[:, 1])
            xyz = np.column_stack([downstream, np.asarray(floor)+1.5])
            arm_air = [main_field(world.config['cavern'], xyz), evaluate_branch_field(world.config, xyz)]
            if not np.any((arm_air[choice] > .25) & (arm_air[1-choice] < -.5)):
                self.availability_reason = 'guide_does_not_enter_selected_exclusive_arm'
                return
            for x in np.arange(nose_x-5., nose_x+11., 1.):
                centers = np.asarray([path[np.argmin(np.abs(path[:, 0]-x))] for path in paths])
                if np.max(np.abs(centers[:, 0]-x)) > .4 or np.linalg.norm(centers[1, :2]-centers[0, :2]) < 8.:
                    continue
                centers[:, 2] += 1.5
                across = np.linspace(centers[0], centers[1], 61)
                rock = np.flatnonzero(world.field(across) < -.25)
                runs = np.split(rock, np.flatnonzero(np.diff(rock) > 1)+1)
                middle = max(runs, key=len)
                if len(middle)*np.linalg.norm(across[1]-across[0]) < 1.5:
                    continue
                offsets = np.array([[0., dy, dz] for dy in (-.6, 0., .6) for dz in (-.45, 0., .45)])
                patches = centers[:, None, :]+offsets[None]
                if np.min(world.field(patches.reshape(-1, 3))) > .15:
                    self.witnesses.append((patches, across[middle[len(middle)//2]]))
            self.chosen = choice
            self.approach_limit_x = nose_x+1.
            self.availability_reason = 'ready' if self.witnesses else 'no_distinct_opening_witnesses'
        except (OSError, ValueError, KeyError, TypeError):
            self.witnesses = []
            self.availability_reason = 'missing_or_invalid_scene_provenance'

    def observe(self, K256, world_from_camera, depth_m, depth_valid):
        result = super().observe(K256, world_from_camera, depth_m, depth_valid)
        if result['fork_visible_verified']:
            result['fork_evidence_criterion'] = 'exclusive_guide_arm_and_two_validated_openings_in_current_rgbd'
        return result
