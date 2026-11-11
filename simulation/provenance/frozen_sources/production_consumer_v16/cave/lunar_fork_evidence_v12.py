"""Conservative fork observations from the current calibrated depth frame."""

import json
from pathlib import Path

import numpy as np


UNKNOWN = dict(fork_visible_verified=False, fork_choice_side='unknown', fork_evidence_criterion='not_verified')
CRITERION = 'two_validated_arm_openings_and_separating_rock_in_current_rgbd'


def _validated_path(world, directory):
    report = json.loads((world.scene / directory / 'obstacle_map_report.json').read_text())
    if not (report['passed'] and report['post_path_geometry']['passed'] and report['source_files_unchanged']):
        raise ValueError('Fork path report is not validated')
    # Reports may have been relocated with a package. Verify bytes via the
    # TerrainWorld hashes instead of requiring the original absolute paths.
    hashes = {Path(name).name: value for name, value in report['source_hashes'].items()}
    if not world.source_hashes or any(hashes.get(name) != value for name, value in world.source_hashes.items()):
        raise ValueError('Fork report does not match the current scene')
    points = np.asarray(report['path_xyz_m'], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Invalid validated fork path')
    return points


class ForkEvidence:
    """Precompute two openings and use each current frame to verify visibility.

    World witnesses and planned arm identity remain private to the collector.
    A route that stops before it reaches the selected arm cannot yield a choice.
    """

    def __init__(self, world, target_id, guide_xy):
        self.witnesses = []
        self.chosen = None
        try:
            paths = [_validated_path(world, directory) for directory in ('validation', 'validation_branch')]
            choice = {'passage_A': 0, 'passage_B': 1}.get(world.targets[target_id]['placement_region'])
            nose_x = float(world.config['junction_v11']['support_center_xy_m'][0])
            guide = np.asarray(guide_xy, dtype=float)
            if choice is None or guide.ndim != 2 or guide.shape[1] < 2 or not np.isfinite(guide).all():
                return
            guide = guide[:, :2]
            # Do not attach a fork decision to a route ending in the common
            # trunk. Require a point on the chosen arm distinct from the other.
            downstream = guide[guide[:, 0] >= nose_x - 12.]
            if not len(downstream):
                return
            distances = [np.linalg.norm(downstream[:, None] - path[None, :, :2], axis=2).min(1) for path in paths]
            if not np.any((distances[choice] <= 4.) & (distances[1-choice] - distances[choice] >= 6.)):
                return

            for x in np.arange(nose_x - 5., nose_x + 11., 1.):
                centers = np.asarray([path[np.argmin(np.abs(path[:, 0] - x))] for path in paths])
                if np.max(np.abs(centers[:, 0] - x)) > .4 or np.linalg.norm(np.diff(centers[:, :2], axis=0)) < 8.:
                    continue
                centers[:, 2] += 1.5
                across = np.linspace(centers[0], centers[1], 61)
                rock = np.flatnonzero(world.field(across) < -.25)
                # A solid contiguous septum is required, not two arbitrary
                # rays through the same opening.
                runs = np.split(rock, np.flatnonzero(np.diff(rock) > 1) + 1)
                middle = max(runs, key=len)
                if len(middle) * np.linalg.norm(across[1] - across[0]) < 1.5:
                    continue
                separator = across[middle[len(middle) // 2]]
                offsets = np.array([[0., dy, dz] for dy in (-.6, 0., .6) for dz in (-.45, 0., .45)])
                patches = centers[:, None, :] + offsets[None, :, :]
                if np.min(world.field(patches.reshape(-1, 3))) > .15:
                    self.witnesses.append((patches, separator))
            self.chosen = choice
            self.approach_limit_x = nose_x + 1.
        except (OSError, ValueError, KeyError, TypeError):
            # Missing map provenance is absence of evidence, never a positive.
            self.witnesses = []

    def observe(self, K256, world_from_camera, depth_m, depth_valid):
        """Return only the verified camera-relative result. Depth is axial Z."""
        if not self.witnesses:
            return dict(UNKNOWN)
        K = np.asarray(K256, dtype=float)
        T = np.asarray(world_from_camera, dtype=float)
        depth = np.asarray(depth_m)
        valid = np.asarray(depth_valid, dtype=bool)
        if K.shape != (3, 3) or T.shape != (4, 4) or depth.ndim != 2 or valid.shape != depth.shape:
            return dict(UNKNOWN)
        if not np.isfinite(K).all() or not np.isfinite(T).all() or T[0, 3] > self.approach_limit_x:
            return dict(UNKNOWN)
        height, width = depth.shape

        def project(points):
            camera = (points - T[:3, 3]) @ T[:3, :3]
            z = camera[:, 2]
            homogeneous = camera @ K.T
            screen = homogeneous[:, :2] / np.maximum(z[:, None], 1e-6)
            return screen, z

        def measured(pixel):
            u, v = np.floor(pixel + .5).astype(int)
            if not (3 <= u < width - 3 and 3 <= v < height - 3):
                return np.empty(0)
            patch = depth[v-1:v+2, u-1:u+2]
            ok = valid[v-1:v+2, u-1:u+2] & np.isfinite(patch) & (patch > 0.)
            return patch[ok]

        for patches, separator in self.witnesses:
            screen, z = project(np.vstack([patches.reshape(-1, 3), separator]))
            if not np.all((z > 2.) & (z < 32.)):
                continue
            pixels = []
            for indices in (range(9), range(9, 18)):
                seen = []
                for i in indices:
                    actual = measured(screen[i])
                    if len(actual) >= 7 and np.min(actual) > z[i] + .2:
                        seen.append(screen[i])
                if len(seen) >= 4 and np.ptp(np.asarray(seen)[:, 0]) >= 2.:
                    pixels.append(float(np.median(np.asarray(seen)[:, 0])))
            if len(pixels) != 2 or abs(pixels[0] - pixels[1]) < .105 * width:
                continue
            middle_u = screen[-1, 0]
            actual = measured(screen[-1])
            if not (min(pixels) + 3. < middle_u < max(pixels) - 3. and len(actual) >= 7
                    and np.median(actual) < z[-1] - .1):
                continue
            side = 'left' if pixels[self.chosen] < pixels[1-self.chosen] else 'right'
            return dict(fork_visible_verified=True, fork_choice_side=side, fork_evidence_criterion=CRITERION)
        return dict(UNKNOWN)
