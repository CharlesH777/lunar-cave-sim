"""Deterministic, collision-aware navigation for the cave search demo.

This is a kinematic base, not a wheel or leg dynamics simulator. Sensors are
rendered by the caller. Ground-truth target positions stay in ``info`` and the
expert; the observation contains the instruction and the agent pose only.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable

import numpy as np


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


class SearchTask:
    """Gymnasium-shaped search task with selectable traditional demonstrations.

    Targets are dictionaries with ``id``, ``position`` (metres), and optionally
    ``instruction``. ``free_space(x,y,z)`` tests cave/obstacle clearance; a 2D
    callable is also accepted. ``floor_height(x,y)`` supplies ground elevation.
    Action = [forward metres/second, yaw radians/second, stop probability].
    """

    def __init__(
        self,
        targets: Iterable[dict],
        free_space: Callable,
        floor_height: Callable = lambda x, y: 0.0,
        seed: int = 7,
        start_pose=(2.0, 0.0, 0.0),
        bounds=(0.0, 29.0, -8.0, 8.0),
        grid_resolution: float = 0.2,
        dt: float = 0.2,
        max_steps: int = 2000,
        radius_m: float = 0.50,
        planning_margin_m: float = 0.12,
        planner: str = 'astar',
    ):
        self.targets = [dict(t) for t in targets]
        if not self.targets:
            raise ValueError("At least one search target is required")
        ids = [t["id"] for t in self.targets]
        if len(set(ids)) != len(ids):
            raise ValueError("Target ids must be unique")
        for t in self.targets:
            p = np.asarray(t["position"], dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError("Target position must be finite xyz")
        if dt <= 0 or grid_resolution <= 0 or max_steps <= 0 or radius_m <= 0:
            raise ValueError("dt, grid_resolution, max_steps and radius_m must be positive")
        if planning_margin_m < 0:
            raise ValueError("planning_margin_m cannot be negative")
        if planner not in ('astar','hybrid'):
            raise ValueError("planner must be astar or hybrid")
        self.planner=planner
        self.path_poses=None
        self.planning_diagnostics={}
        self._smooth_follower=None
        self.free_space = free_space
        self.floor_height = floor_height
        self.rng = np.random.default_rng(seed)
        self.start_pose = np.asarray(start_pose, dtype=float)
        if self.start_pose.shape != (3,) or not np.isfinite(self.start_pose).all():
            raise ValueError("start_pose must be finite x,y,yaw")
        self.bounds = tuple(map(float, bounds))
        self.resolution, self.dt = float(grid_resolution), float(dt)
        self.max_steps = int(max_steps)
        self.radius = float(radius_m)
        self.planning_margin = float(planning_margin_m)
        self._grid_cache = {}
        self._path = None
        self.pose = self.start_pose.copy()
        self.target = self.targets[0]
        self.steps = 0
        self.path_length = 0.0
        self.collisions = 0
        self.done = False
        angles = np.arange(16) * (2 * math.pi / 16)
        self._footprint = [(0.0, 0.0)] + list(zip(
            self.radius * np.cos(angles), self.radius * np.sin(angles)
        ))
        # A grid path is followed by a finite-rate heading controller. Reserve
        # clearance around that path so rounding and corner tracking cannot push
        # the physical footprint into a wall that the ideal centreline grazes.
        self._planning_footprint = self._footprint + list(zip(
            (self.radius + self.planning_margin) * np.cos(angles),
            (self.radius + self.planning_margin) * np.sin(angles),
        ))

    def _inside(self, x, y, z):
        try:
            return bool(self.free_space(float(x), float(y), float(z)))
        except TypeError:
            return bool(self.free_space(float(x), float(y)))

    def walkable(self, x: float, y: float, *, planning: bool = False) -> bool:
        """Check the base's circular footprint at two body heights."""
        xmin, xmax, ymin, ymax = self.bounds
        radius = self.radius + (self.planning_margin if planning else 0.0)
        if not (xmin + radius <= x <= xmax - radius and
                ymin + radius <= y <= ymax - radius):
            return False
        base = float(self.floor_height(x, y))
        if not math.isfinite(base):
            return False
        for dx, dy in self._planning_footprint if planning else self._footprint:
            px, py = x + dx, y + dy
            floor = float(self.floor_height(px, py))
            # Reject a step/ledge taller than the base can negotiate.
            if not math.isfinite(floor) or abs(floor - base) > 0.18:
                return False
            if not (self._inside(px, py, base + 0.3) and
                    self._inside(px, py, base + 1.0)):
                return False
        return True

    def _clear_segment(self, a, b, *, planning: bool = False) -> bool:
        distance = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
        # Include an interior sample even for each short integration segment;
        # endpoint-only checks can skip a thin wall between footprint samples.
        count = max(2, math.ceil(distance / 0.02))
        return all(self.walkable(*(np.asarray(a) + (np.asarray(b) - a) * f), planning=planning)
                   for f in np.linspace(0.0, 1.0, count + 1))

    def _observation(self):
        x, y, yaw = self.pose
        return {
            "instruction": self.target.get("instruction", f"Find the {self.target['id']}"),
            "pose": [float(x), float(y), float(self.floor_height(x, y)), float(yaw)],
            "step": self.steps,
        }

    def reset(self, target_id=None, *, seed=None, start_pose=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if target_id is None:
            self.target = self.targets[int(self.rng.integers(len(self.targets)))]
        else:
            matches = [t for t in self.targets if t["id"] == target_id]
            if not matches:
                raise KeyError(f"Unknown target: {target_id}")
            self.target = matches[0]
        self.pose = np.asarray(self.start_pose if start_pose is None else start_pose, dtype=float).copy()
        if self.pose.shape != (3,) or not np.isfinite(self.pose).all():
            raise ValueError("start_pose must be finite x,y,yaw")
        self.pose[2] = wrap_angle(self.pose[2])
        if not self.walkable(*self.pose[:2]):
            raise ValueError(f"Start pose is not collision free: {self.pose.tolist()}")
        self.steps, self.path_length, self.collisions = 0, 0.0, 0
        self.done = False
        self._path = None
        self.path_poses=None
        self.planning_diagnostics={}
        self._smooth_follower=None
        # Occupancy may change when the caller randomizes props between resets.
        self._grid_cache.clear()
        self.initial_distance = self._distance()
        return self._observation()

    def _distance(self):
        return float(np.linalg.norm(self.pose[:2] - np.asarray(self.target["position"][:2])))

    def step(self, action, *, visible_pixel_count: int = 0):
        """Advance one control interval and evaluate a stop with real visibility.

        ``visible_pixel_count`` must come from the current target's rendered
        instance mask, before this action. It is used only when action[2]>=0.5;
        stop never moves the agent. Absence of sensor evidence cannot succeed.
        """
        if self.done:
            raise RuntimeError("Episode ended; call reset()")
        a = np.asarray(action, dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError("Action must contain three finite values")
        before = self._distance()
        collision = False
        stop = bool(a[2] >= 0.5)
        if not stop:
            v = float(np.clip(a[0], -0.6, 0.6))
            omega = float(np.clip(a[1], -1.2, 1.2))
            # Integrate the differential-drive arc in short, swept increments.
            n = max(1, math.ceil(abs(v * self.dt) / 0.04))
            interval = self.dt / n
            for _ in range(n):
                heading = self.pose[2] + omega * interval / 2
                new_xy = self.pose[:2] + v * interval * np.array([math.cos(heading), math.sin(heading)])
                if self._clear_segment(self.pose[:2], new_xy):
                    self.path_length += float(np.linalg.norm(new_xy - self.pose[:2]))
                    self.pose[:2] = new_xy
                else:
                    collision = True
                self.pose[2] = wrap_angle(self.pose[2] + omega * interval)
        self.steps += 1
        self.collisions += int(collision)
        delta = np.asarray(self.target["position"][:2]) - self.pose[:2]
        facing_error = abs(wrap_angle(math.atan2(delta[1], delta[0]) - self.pose[2]))
        distance = self._distance()
        visible = int(visible_pixel_count) >= 32
        success = bool(stop and distance <= 1.8 and facing_error <= 0.65 and visible)
        terminated = stop
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self.done = terminated or truncated
        reward = before - distance - 0.01 - 0.1 * collision
        if stop:
            reward += 1.0 if success else -1.0
        info = {
            "target_id": self.target["id"],
            "distance_to_target_m": distance,
            "facing_error_rad": facing_error,
            "visible_pixel_count": int(visible_pixel_count),
            "success": success, "collision": collision,
            "collision_steps": self.collisions, "path_length_m": self.path_length,
            "stop_requested": stop,
            "termination_reason": ("found" if success else "incorrect_stop") if stop else
                                  ("step_limit" if truncated else None),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _cell(self, xy):
        return (int(round((xy[0] - self.bounds[0]) / self.resolution)),
                int(round((xy[1] - self.bounds[2]) / self.resolution)))

    def _xy(self, cell):
        return np.array([self.bounds[0] + cell[0] * self.resolution,
                         self.bounds[2] + cell[1] * self.resolution])

    def _cell_free(self, cell):
        if cell not in self._grid_cache:
            self._grid_cache[cell] = self.walkable(*self._xy(cell), planning=True)
        return self._grid_cache[cell]

    def _target_line_of_sight(self, xy):
        start = np.array([*xy, float(self.floor_height(*xy)) + 1.0])
        end = np.asarray(self.target["position"], dtype=float)
        delta = end - start
        # The target is itself solid. Test intervening geometry only, stopping
        # just before its conservative collision radius rather than at its centre.
        stand_off = max(0.0, float(self.target.get("radius", 0.2))) + 0.08
        # The supplied target radius describes an XY collision footprint, so
        # compensate for the downward camera ray rather than cutting inside a
        # low object's cylindrical collision envelope.
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        if horizontal_distance <= stand_off:
            return True
        end = start + delta * ((horizontal_distance - stand_off) / horizontal_distance)
        # Geometry-level planning aid only. Success still requires the mask.
        return all(self._inside(*p) for p in np.linspace(start, end, max(3, int(np.linalg.norm(end-start)/0.08))))

    def plan_path(self):
        """Plan using the requested method; never silently fall back."""
        if self.planner=='astar':
            path=self._plan_astar_path()
            self.planning_diagnostics={'planner':'grid_astar','tracking':'heading_P_turn_then_drive'}
            return path
        from .hybrid_planner import plan_hybrid
        from .path_tracking import SmoothPathFollower
        poses,diagnostics=plan_hybrid(self)
        self.path_poses=np.asarray(poses,dtype=float)
        if self.path_poses.ndim!=2 or self.path_poses.shape[1]!=3:
            raise RuntimeError('Hybrid planner must return Nx3 SE(2) poses')
        self.planning_diagnostics=dict(diagnostics)
        self._path=[p.copy() for p in self.path_poses[1:,:2]]
        self._smooth_follower=SmoothPathFollower(self,self.path_poses)
        return self.path_poses[:,:2].copy()

    @property
    def controller_diagnostics(self):
        return dict(self._smooth_follower.diagnostics) if self._smooth_follower is not None else {}

    def _plan_astar_path(self):
        """A* on a 20 cm grid, stopping about a metre from the target."""
        start = self._cell(self.pose[:2])
        if not self._cell_free(start) or not self._clear_segment(self.pose[:2], self._xy(start), planning=True):
            raise RuntimeError("Cannot snap agent to a collision-free planning cell")
        goal = np.asarray(self.target["position"][:2], dtype=float)
        def h(cell):
            return max(0.0, float(np.linalg.norm(self._xy(cell) - goal)) - 1.2)
        queue, costs, parent = [(h(start), 0.0, start)], {start: 0.0}, {}
        final = None
        directions = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        while queue:
            _, g, cell = heapq.heappop(queue)
            if g > costs[cell] + 1e-9:
                continue
            xy = self._xy(cell)
            if 0.75 <= np.linalg.norm(xy - goal) <= 1.2 and self._target_line_of_sight(xy):
                final = cell
                break
            for dx, dy in directions:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not self._cell_free(nxt):
                    continue
                if dx and dy and not (self._cell_free((cell[0]+dx, cell[1])) and
                                      self._cell_free((cell[0], cell[1]+dy))):
                    continue
                # Check continuous edge as well as grid points.
                if not self._clear_segment(xy, self._xy(nxt), planning=True):
                    continue
                ng = g + self.resolution * math.hypot(dx, dy)
                if ng < costs.get(nxt, math.inf):
                    costs[nxt], parent[nxt] = ng, cell
                    heapq.heappush(queue, (ng + h(nxt), ng, nxt))
        if final is None:
            raise RuntimeError(f"No reachable viewing position for target {self.target['id']}")
        cells = [final]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        points = [self.pose[:2].copy()] + [self._xy(c) for c in reversed(cells)]
        # Greedy shortcutting retains the same swept-clearance contract.
        path, i = [points[0]], 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self._clear_segment(points[i], points[j], planning=True):
                j -= 1
            path.append(points[j])
            i = j
        self._path = path[1:]
        return np.asarray(path)

    def expert_action(self):
        """Selected ground-truth traditional expert; not a learned VLA policy."""
        if self.done:
            raise RuntimeError("Episode ended; call reset()")
        if self.planner=='hybrid':
            if self._smooth_follower is None:self.plan_path()
            return self._smooth_follower.action()
        if self._path is None:
            self.plan_path()
        while self._path and np.linalg.norm(self._path[0] - self.pose[:2]) < 0.025:
            self._path.pop(0)
        target_xy = np.asarray(self.target["position"][:2], dtype=float)
        aim = self._path[0] if self._path else target_xy
        delta = aim - self.pose[:2]
        error = wrap_angle(math.atan2(delta[1], delta[0]) - self.pose[2])
        if not self._path and abs(error) < 0.10:
            return np.array([0.0, 0.0, 1.0])
        omega = float(np.clip(3.0 * error, -1.2, 1.2))
        speed = 0.0 if abs(error) > 0.015 or not self._path else min(0.6, np.linalg.norm(delta) / self.dt)
        return np.array([speed, omega, 0.0])

    def expert_actions(self, max_steps=None):
        """Yield actions; caller must render sensors and call step after each."""
        for _ in range(self.max_steps if max_steps is None else max_steps):
            if self.done:
                return
            action = self.expert_action()
            yield action
            if action[2] >= 0.5:
                return
