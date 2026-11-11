"""Regulated pure pursuit for a collision-checked differential-drive base.

The supplied SE(2) path is followed in local arc-length order.  Commands include
velocity/turn-rate acceleration limits and a short-horizon plus full-braking
collision check.  No physics state is changed by this module; the caller must
apply each returned action once through SearchTask.step().
"""
from __future__ import annotations

import math
import numpy as np


def _wrap(a):
    return (float(a) + math.pi) % (2 * math.pi) - math.pi


class SmoothPathFollower:
    """Track ordered Nx3 ``x,y,yaw`` poses using regulated pure pursuit.

    ``action()`` is called exactly once before each task.step().  A safety brake
    is reported in ``diagnostics``; it is never recorded as a collision-free
    teleport or an implicit successful stop.  Only SearchTask's rendered-mask
    gate can turn the requested STOP into a successful search result.
    """

    def __init__(self, task, path_poses, *, max_linear_accel=.4,
                 max_angular_accel=1.8, max_speed=.6, max_omega=1.2,
                 min_lookahead_m=.40, max_lookahead_m=.80,
                 lookahead_time_s=.65):
        path = np.asarray(path_poses, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all():
            raise ValueError('path_poses must contain at least two finite x,y,yaw poses')
        if min(max_linear_accel, max_angular_accel, max_speed, max_omega) <= 0:
            raise ValueError('Velocity and acceleration limits must be positive')
        if min_lookahead_m <= 0 or max_lookahead_m < min_lookahead_m or lookahead_time_s < 0:
            raise ValueError('Lookahead bounds must be positive and ordered; lookahead time cannot be negative')
        keep = np.r_[True, np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) > 1e-7]
        path = path[keep]
        if len(path) < 2:
            raise ValueError('Path must have nonzero positional length')
        self.task, self.path = task, path.copy()
        self.dt = float(task.dt)
        if self.dt <= 0:
            raise ValueError('Task dt must be positive')
        self.accel, self.angular_accel = float(max_linear_accel), float(max_angular_accel)
        self.max_speed, self.max_omega = float(max_speed), float(max_omega)
        self.min_lookahead = float(min_lookahead_m)
        self.max_lookahead = float(max_lookahead_m)
        self.lookahead_time = float(lookahead_time_s)
        self.segment_lengths = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
        self.s = np.r_[0., np.cumsum(self.segment_lengths)]
        self.total_length = float(self.s[-1])
        self.progress = 0.
        self._segment = 0
        self._v = self._w = 0.
        self._last_step = None
        self._last_action = None
        self.diagnostics = {
            'controller': 'regulated_pure_pursuit_with_acceleration_and_braking_checks',
            'max_linear_accel_mps2': self.accel,
            'max_angular_accel_radps2': self.angular_accel,
            'min_lookahead_m': self.min_lookahead,
            'max_lookahead_m': self.max_lookahead,
            'lookahead_time_s': self.lookahead_time,
            'commands': 0, 'safety_brake_steps': 0, 'blocked_steps': 0,
            'moving_turn_steps': 0, 'stationary_turn_steps': 0,
            'progress_m': 0., 'remaining_m': self.total_length,
            'blocked': False, 'cross_track_error_m': 0., 'lookahead_m': 0.,
        }

    def _projection(self, xy):
        # A local window prevents a crossing or a nearby return gallery from
        # jumping progress to a much later part of a loop.
        start = max(0, self._segment - 2)
        end_s = self.progress + max(.9, self._v * self.dt * 4)
        end = min(len(self.segment_lengths), int(np.searchsorted(self.s, end_s)) + 1)
        best = None
        for i in range(start, max(start + 1, end)):
            delta = self.path[i + 1, :2] - self.path[i, :2]
            t = float(np.clip((xy - self.path[i, :2]) @ delta / (self.segment_lengths[i] ** 2), 0., 1.))
            projected_s = self.s[i] + t * self.segment_lengths[i]
            # Small backward projections are allowed for the distance estimate,
            # but cannot move monotonic progress backwards.
            if projected_s < self.progress - .15:
                continue
            point = self.path[i, :2] + t * delta
            score = float(np.sum((point - xy) ** 2))
            if best is None or score < best[0]:
                best = (score, projected_s, i)
        if best is not None:
            self.progress = max(self.progress, min(float(best[1]), self.total_length))
            self._segment = int(np.searchsorted(self.s, self.progress, side='right') - 1)
            self._segment = min(self._segment, len(self.segment_lengths) - 1)
            return math.sqrt(best[0])
        return float(np.linalg.norm(xy - self._point(self.progress)))

    def _point(self, distance):
        distance = float(np.clip(distance, 0., self.total_length))
        i = min(int(np.searchsorted(self.s, distance, side='right') - 1), len(self.segment_lengths) - 1)
        i = max(0, i)
        t = (distance - self.s[i]) / self.segment_lengths[i]
        return self.path[i, :2] + t * (self.path[i + 1, :2] - self.path[i, :2])

    @staticmethod
    def _integrate(pose, v, w, interval):
        x, y, yaw = pose
        turn = w * interval
        if abs(w) < 1e-8:
            return np.array([x + v * interval * math.cos(yaw),
                             y + v * interval * math.sin(yaw), yaw])
        return np.array([x + v / w * (math.sin(yaw + turn) - math.sin(yaw)),
                         y + v / w * (math.cos(yaw) - math.cos(yaw + turn)),
                         _wrap(yaw + turn)])

    def _limited(self, v, w):
        v = float(np.clip(v, max(0., self._v - self.accel * self.dt),
                          min(self.max_speed, self._v + self.accel * self.dt)))
        w = float(np.clip(w, max(-self.max_omega, self._w - self.angular_accel * self.dt),
                          min(self.max_omega, self._w + self.angular_accel * self.dt)))
        return v, w

    def _rollout_safe(self, v, w):
        # Hold the proposed command for one control period, then follow a full
        # braking manoeuvre at approximately constant curvature.  A command
        # cannot be accepted merely because its next endpoint is still clear.
        pose = np.asarray(self.task.pose, dtype=float).copy()
        if not self.task.walkable(*pose[:2]):
            return False
        elapsed = 0.
        velocity, omega = v, w
        horizon = self.dt + v / self.accel
        while elapsed < horizon - 1e-9:
            interval = min(.05, horizon - elapsed)
            # At most 2cm between collision samples, including a low-speed step.
            interval = min(interval, .02 / max(velocity, .02))
            if elapsed >= self.dt - 1e-9:
                new_velocity = max(0., velocity - self.accel * interval)
                new_omega = omega * new_velocity / velocity if velocity > 1e-7 else 0.
            else:
                interval = min(interval, self.dt - elapsed)
                new_velocity, new_omega = velocity, omega
            pose = self._integrate(pose, (velocity + new_velocity) / 2,
                                   (omega + new_omega) / 2, interval)
            if not self.task.walkable(*pose[:2]):
                return False
            elapsed += interval
            velocity, omega = new_velocity, new_omega
        return True

    def _commit(self, v, w, stop=False, *, blocked=False, lookahead=0., heading_error=0.):
        # The STOP command is issued after both command states have settled.
        self._v, self._w = float(v), float(w)
        self.diagnostics['commands'] += 1
        self.diagnostics['blocked'] = bool(blocked)
        self.diagnostics['blocked_steps'] += int(blocked)
        self.diagnostics['moving_turn_steps'] += int(v > .03 and abs(w) > .03)
        self.diagnostics['stationary_turn_steps'] += int(v <= .03 and abs(w) > .03)
        self.diagnostics.update(progress_m=self.progress,
                                remaining_m=max(0., self.total_length - self.progress),
                                lookahead_m=float(lookahead), heading_error_rad=float(heading_error),
                                command_v_mps=float(v), command_omega_radps=float(w))
        result = np.array([v, w, float(stop)])
        self._last_step = getattr(self.task, 'steps', None)
        self._last_action = result.copy()
        return result

    def action(self):
        """Return one acceleration-bounded [linear speed,yaw speed,STOP]."""
        if getattr(self.task, 'done', False):
            raise RuntimeError('Episode ended; create a follower after reset')
        if self._last_step is not None and self._last_step == getattr(self.task, 'steps', None):
            return self._last_action.copy()
        pose = np.asarray(self.task.pose, dtype=float)
        xy, yaw = pose[:2], float(pose[2])
        cross_track = self._projection(xy)
        self.diagnostics['cross_track_error_m'] = cross_track
        remaining = max(0., self.total_length - self.progress)
        endpoint_distance = float(np.linalg.norm(self.path[-1, :2] - xy))
        target_delta = np.asarray(self.task.target['position'][:2]) - xy
        target_distance = float(np.linalg.norm(target_delta))
        target_error = _wrap(math.atan2(target_delta[1], target_delta[0]) - yaw)

        # Positioning and final camera alignment are separate from the normal
        # curved-path tracking.  Rotation in place is limited to the endpoint
        # (or a large initial heading mismatch), rather than every path node.
        at_endpoint = remaining < .13 and endpoint_distance < .14
        if at_endpoint:
            desired_w = 0. if abs(target_error) < .06 else float(np.clip(2.2 * target_error, -self.max_omega, self.max_omega))
            v, w = self._limited(0., desired_w)
            if (self._v < .008 and abs(self._w) < .035 and v < .008 and abs(w) < .035
                    and abs(target_error) < .10 and target_distance <= 1.8):
                return self._commit(0., 0., True, heading_error=target_error)
            if self._rollout_safe(v, w):
                return self._commit(v, w, heading_error=target_error)

        lookahead = float(np.clip(self.min_lookahead + self.lookahead_time * self._v,
                                  self.min_lookahead, self.max_lookahead))
        for distance in (lookahead, max(.24, lookahead * .72), .22):
            look = self._point(self.progress + distance)
            delta = look - xy
            local_y = -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1]
            heading_error = _wrap(math.atan2(delta[1], delta[0]) - yaw)
            curvature = 2 * local_y / max(float(delta @ delta), .025)
            path_to_go = max(remaining, endpoint_distance if remaining < .35 else 0.)
            speed = min(self.max_speed, math.sqrt(.35 / max(abs(curvature), .01)),
                        math.sqrt(2 * self.accel * max(0., path_to_go - .025)),
                        max(0., .95 * path_to_go))
            # Continue moving through normal bends; a severely mismatched start
            # first aligns safely instead of trying a high-curvature forward arc.
            speed *= max(.15, math.cos(min(abs(heading_error), math.pi/2)))
            if abs(heading_error) > 1.05:
                speed = 0.
            v, _ = self._limited(speed, 0.)
            desired_w = curvature * v if speed > .001 else 2.2 * heading_error
            v, w = self._limited(speed, float(np.clip(desired_w, -self.max_omega, self.max_omega)))
            if self._rollout_safe(v, w):
                return self._commit(v, w, lookahead=distance, heading_error=heading_error)

        # Static-scene feasibility is protected by the preceding full-braking
        # checks. If tracking cannot advance, decelerate along its existing arc
        # and report a blocked state explicitly for the caller to replan.
        brake_v = max(0., self._v - self.accel * self.dt)
        brake_w = self._w * brake_v / self._v if self._v > 1e-7 else 0.
        brake_v, brake_w = self._limited(brake_v, brake_w)
        self.diagnostics['safety_brake_steps'] += 1
        if not self._rollout_safe(brake_v, brake_w):
            raise RuntimeError('No collision-free acceleration-bounded braking command; scene changed inside braking distance')
        return self._commit(brake_v, brake_w, blocked=True, lookahead=lookahead)
