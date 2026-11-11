"""Compare legacy A*+P and Hybrid A*+regulated pursuit on saved cave scenes.

CPU-only control evaluation. The target-mask count is explicitly injected, so
success here never constitutes rendered visual evidence or learned VLA results.
No simulator, GPU, map mutation, or new dependency installation is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from cave.lunar_geometry import cave_field, configure_layout, floor_height
from cave.navigation import SearchTask


ROOT = Path(__file__).resolve().parent
REFERENCE_LINKS = {
    "hybrid_astar": "https://ros-navigation.github.io/mkdocs.nav2.org/jazzy/configuration_and_development/configuration_guide/planners_plugins/smac/smac_hybrid/configuring_smac_hybrid/",
    "regulated_pursuit": "https://github.com/ros-navigation/navigation2/blob/main/nav2_regulated_pure_pursuit_controller/README.md",
    "regulated_pursuit_paper": "https://arxiv.org/abs/2305.20026",
}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    names = ["cave/navigation.py", "cave/hybrid_planner.py", "cave/path_tracking.py",
             "cave/lunar_geometry.py", "cave/layouts.py", Path(__file__).name]
    return {name: sha256(ROOT / name) for name in names if (ROOT / name).is_file()}


def make_task(manifest, planner, max_steps):
    """Use precisely the quantized collision adapter in run_demo.py."""
    obstacles = manifest["obstacles"]

    @lru_cache(maxsize=250000)
    def free_cached(ix, iy, iz):
        x, y, z = ix * .04, iy * .04, iz * .04
        if cave_field(x, y, z) < .035:
            return False
        for obstacle in obstacles:
            if (obstacle.get("kind") == "pool" or z <= obstacle["height"] + .05) and (
                (x - obstacle["x"]) ** 2 + (y - obstacle["y"]) ** 2 < (obstacle["radius"] + .04) ** 2
            ):
                return False
        return True

    def free(x, y, z):
        return free_cached(round(x / .04), round(y / .04), round(z / .04))

    return SearchTask(
        manifest["targets"], free_space=free, floor_height=floor_height,
        seed=manifest.get("seed", 7), start_pose=tuple(manifest.get("spawn", [2., .3, 0.])),
        bounds=tuple(manifest.get("layout", {}).get("bounds", [-.5, 34., -14., 14.])),
        grid_resolution=.2, dt=.2, max_steps=max_steps,
        radius_m=.50, planning_margin_m=.12, planner=planner,
    )


def absolute_stat(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"max": 0., "mean_abs": 0., "rms": 0.}
    return {"max": float(np.max(np.abs(values))), "mean_abs": float(np.mean(np.abs(values))),
            "rms": float(np.sqrt(np.mean(values * values)))}


def cross_track_distance(points, reference):
    """Exact nearest distance to reference segments, independent of point density."""
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(reference) < 2:
        return np.full(len(points), np.nan)
    a, delta = reference[:-1, :2], np.diff(reference[:, :2], axis=0)
    lengths2 = np.einsum("ij,ij->i", delta, delta)
    keep = lengths2 > 1e-12
    a, delta, lengths2 = a[keep], delta[keep], lengths2[keep]
    if not len(a):
        return np.linalg.norm(points[:, :2] - reference[0, :2], axis=1)
    offset = points[:, None, :2] - a[None, :, :]
    t = np.clip(np.einsum("nsi,si->ns", offset, delta) / lengths2, 0., 1.)
    residual = offset - t[..., None] * delta[None, :, :]
    return np.sqrt(np.min(np.einsum("nsi,nsi->ns", residual, residual), axis=1))


def trajectory_metrics(poses, commands, reference, dt, info):
    poses = np.asarray(poses, dtype=float)
    commands = np.asarray(commands, dtype=float).reshape(-1, 3)
    if not len(commands):
        return {}
    v, omega = commands[:, 0], commands[:, 1]
    derivative = np.diff(np.vstack([np.zeros((1, 2)), commands[:, :2]]), axis=0) / dt
    moving = np.abs(v) > .01
    rotating = (np.abs(v) <= .01) & (np.abs(omega) > .01)
    yaw_delta = np.diff(np.unwrap(poses[:, 2]))
    distances = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    # Rotation in place is separately counted; division by near-zero progress
    # would turn its undefined translational curvature into a misleading number.
    progress = distances > .005
    curvature = yaw_delta[progress] / distances[progress]
    tracking_error = cross_track_distance(poses[:, :2], reference)
    return {
        "control_intervals": len(commands), "simulated_time_s": float(len(commands) * dt),
        "path_length_m": float(info.get("path_length_m", distances.sum())),
        "sampled_pose_path_length_m": float(distances.sum()),
        "planned_path_length_m": float(np.linalg.norm(np.diff(reference[:, :2], axis=0), axis=1).sum()),
        "moving_fraction": float(moving.mean()),
        "rotating_in_place_fraction": float(rotating.mean()),
        "rotating_in_place_intervals": int(rotating.sum()),
        "stationary_fraction": float((~moving & ~rotating).mean()),
        "heading_total_variation_rad": float(np.abs(yaw_delta).sum()),
        "angular_velocity_total_variation_radps": float(np.abs(np.diff(np.r_[0., omega])).sum()),
        "linear_velocity_total_variation_mps": float(np.abs(np.diff(np.r_[0., v])).sum()),
        "linear_speed_mps": absolute_stat(v), "yaw_speed_radps": absolute_stat(omega),
        "linear_acceleration_mps2": absolute_stat(derivative[:, 0]),
        "angular_acceleration_radps2": absolute_stat(derivative[:, 1]),
        "lateral_acceleration_v_times_omega_mps2": absolute_stat(v * omega),
        "moving_curvature_per_m": absolute_stat(curvature),
        "cross_track_error_m": absolute_stat(tracking_error[np.isfinite(tracking_error)]),
    }


def baseline_parity(input_folder, commands, poses, result, dt):
    """Guard against accidentally changing the old controller during comparison."""
    transitions = input_folder / "episode_000/transitions.jsonl"
    result_file = input_folder / "episode_000/result.json"
    if not transitions.is_file() or not result_file.is_file():
        return {"available": False}
    rows = [json.loads(line) for line in transitions.read_text().splitlines() if line.strip()]
    saved_actions = np.asarray([row["action"] for row in rows], dtype=float)
    saved_poses = np.asarray([[row["observation"]["pose"][i] for i in [0, 1, 3]] for row in rows], dtype=float)
    saved_result = json.loads(result_file.read_text())
    same_count = len(saved_actions) == len(commands)
    command_error = float(np.max(np.abs(saved_actions - commands))) if same_count else None
    pose_error = float(np.max(np.abs(saved_poses - poses[:-1]))) if same_count else None
    length_delta = abs(float(saved_result["path_length_m"]) - float(result.get("path_length_m", math.inf)))
    return {
        "available": True, "reference_kind": "saved actual RTX-rendered narrow-map expert episode",
        "reference_frames": len(rows), "same_control_count": same_count,
        "max_command_abs_difference": command_error, "max_pose_abs_difference": pose_error,
        "path_length_abs_difference_m": length_delta,
        "matches_saved_rendered_baseline": bool(same_count and command_error < 1e-9 and pose_error < 1e-9 and length_delta < 1e-8),
        "transitions_sha256": sha256(transitions), "result_sha256": sha256(result_file), "dt_s": dt,
    }


def optional_diagnostics(env):
    result = {}
    for name in ["planning_diagnostics", "planner_diagnostics", "controller_diagnostics"]:
        value = getattr(env, name, None)
        if value is not None:
            result[name] = plain(value() if callable(value) else value)
    for name in ["_follower", "follower"]:
        follower = getattr(env, name, None)
        if follower is not None:
            for key in ["diagnostics", "blocked", "blocked_reason", "config"]:
                value = getattr(follower, key, None)
                if value is not None:
                    result["follower_" + key] = plain(value() if callable(value) else value)
    return result


def run_case(map_id, manifest, mode, target, max_steps, folder, input_folder):
    configure_layout(map_id)
    report = {"planner": mode, "cpu_visible_pixels_injected": 64, "success": False,
              "sensor_scope": "synthetic 64-pixel evidence for control testing only; no renderer invoked"}
    poses, commands, infos, path = [], [], [], np.empty((0, 2))
    env = make_task(manifest, mode, max_steps)
    started = time.perf_counter()
    try:
        env.reset(target)
        poses.append(env.pose.copy())
        planning_start = time.perf_counter()
        path = np.asarray(env.plan_path(), dtype=float)
        report["planning_seconds"] = time.perf_counter() - planning_start
        for _ in range(max_steps):
            command = np.asarray(env.expert_action(), dtype=float)
            _, reward, terminated, truncated, info = env.step(command, visible_pixel_count=64)
            commands.append(command.copy())
            poses.append(env.pose.copy())
            infos.append({**info, "reward": reward, "terminated": terminated, "truncated": truncated})
            if terminated or truncated:
                break
        report.update(infos[-1] if infos else {})
        report["success"] = bool(infos and infos[-1]["success"])
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc(limit=8)}
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        p = np.asarray(poses, dtype=float).reshape(-1, 3)
        c = np.asarray(commands, dtype=float).reshape(-1, 3)
        report["metrics"] = trajectory_metrics(p, c, path, env.dt, infos[-1] if infos else {})
        report["diagnostics"] = optional_diagnostics(env)
        if mode == "astar" and len(commands):
            report["saved_baseline_parity"] = baseline_parity(input_folder, c, p, infos[-1], env.dt)
        archive = folder / f"map_{map_id}_{mode}.npz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"pose_xy_yaw": p, "commands_v_omega_stop": c, "planned_xy": path[:, :2],
                  "dt_s": np.array(env.dt)}
        for name in ["path_poses", "planned_poses", "_path_poses"]:
            detailed = getattr(env, name, None)
            if detailed is not None:
                arrays["planned_poses_xy_yaw"] = np.asarray(detailed, dtype=float)
                break
        np.savez_compressed(archive, **arrays)
        report["trajectory_file"] = str(archive)
        report["trajectory_sha256"] = sha256(archive)
        atomic_json(folder / f"map_{map_id}_{mode}.json", report)
    return report


def plot_map(folder, map_id, manifest, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_layout(map_id)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    xx, yy = np.meshgrid(np.arange(-.5, 34., .12), np.arange(-14., 14., .12))
    field = cave_field(xx, yy, floor_height(xx, yy) + .65)
    axes[0, 0].contourf(xx, yy, field, levels=[.035, 100.], colors=["#e0e4e6"])
    axes[0, 0].contour(xx, yy, field, levels=[.035], colors=["#9da6ab"], linewidths=.6)
    colors = {"astar": "#d95f02", "hybrid": "#007c91"}
    for mode, case in cases.items():
        archive = np.load(case["trajectory_file"], allow_pickle=False)
        poses, commands = archive["pose_xy_yaw"], archive["commands_v_omega_stop"]
        if not len(commands):
            continue
        t = np.arange(len(commands)) * float(archive["dt_s"])
        path = archive["planned_xy"]
        if len(path):
            axes[0, 0].plot(path[:, 0], path[:, 1], color=colors[mode], linestyle="--", alpha=.45, linewidth=1)
        axes[0, 0].plot(poses[:, 0], poses[:, 1], color=colors[mode], label=mode, linewidth=1.7)
        axes[0, 1].plot(t, commands[:, 0], color=colors[mode], label=mode)
        axes[1, 0].plot(t, commands[:, 1], color=colors[mode], label=mode)
        accel = np.diff(np.r_[0., commands[:, 1]]) / float(archive["dt_s"])
        axes[1, 1].plot(t, accel, color=colors[mode], label=mode)
    axes[0, 0].set(xlabel="x / m", ylabel="y / m", title="Executed path (dashed: plan)")
    axes[0, 0].set_aspect("equal")
    axes[0, 1].set(xlabel="time / s", ylabel="v / m s⁻¹", title="Linear velocity")
    axes[1, 0].set(xlabel="time / s", ylabel="ω / rad s⁻¹", title="Yaw velocity")
    axes[1, 1].set(xlabel="time / s", ylabel="dω/dt / rad s⁻²", title="Yaw acceleration")
    for axis in axes.ravel():
        axis.grid(alpha=.2)
        axis.legend()
    fig.suptitle(f"Map {map_id}: {manifest.get('map_name', '')}\nCPU controls; injected target mask; identical saved scene and vehicle footprint")
    fig.savefig(folder / f"map_{map_id}_navigation_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", nargs="+", default=["01"])
    parser.add_argument("--all", action="store_true", help="Evaluate all eight saved narrow layouts")
    parser.add_argument("--modes", nargs="+", choices=["astar", "hybrid"], default=["astar", "hybrid"])
    parser.add_argument("--target", default="first_aid_kit")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs/maps_revised")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/maps_smooth")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if "planner" not in inspect.signature(SearchTask).parameters:
        raise SystemExit("SearchTask planner='astar'|'hybrid' API is not available yet; benchmark not run")
    maps = [f"{i:02d}" for i in range(1, 9)] if args.all else [str(m).zfill(2) for m in args.maps]
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "phase": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "CPU paired saved-scene control comparison with injected64 mask; no GPU, rendered visibility, or trained VLA claim",
              "input_root": str(args.input_root.resolve()), "target_id": args.target,
              "vehicle": {"physical_radius_m": .50, "planning_margin_m": .12, "dt_s": .2,
                          "max_linear_mps": .6, "max_yaw_radps": 1.2},
              "metric_notes": {"derivatives": "Discrete commanded accelerations include the initial zero-velocity boundary and terminal STOP command",
                               "curvature": "Realized pose yaw change / translated distance for steps progressing >5mm; in-place turns reported separately",
                               "path_tracking_error": "Distance to exact planned XY segments, independent of their point density"},
              "references": REFERENCE_LINKS, "source_sha256": source_hashes(), "maps": {}}
    destination = out / "navigation_comparison.json"
    atomic_json(destination, report)
    for map_id in maps:
        folder = args.input_root / f"map_{map_id}"
        manifest_file = folder / "scene_manifest.json"
        manifest = json.loads(manifest_file.read_text())
        cases = {}
        report["current_map"] = map_id
        entry = {"manifest_sha256": sha256(manifest_file), "map_name": manifest.get("map_name"),
                 "revision": manifest.get("revision"), "obstacle_count": len(manifest["obstacles"]), "cases": cases}
        report["maps"][map_id] = entry
        for mode in args.modes:
            print(f"BENCHMARK map={map_id} mode={mode}", flush=True)
            cases[mode] = run_case(map_id, manifest, mode, args.target, args.max_steps, out, folder)
            atomic_json(destination, report)
            metrics = cases[mode].get("metrics", {})
            print(json.dumps({"map": map_id, "mode": mode, "success": cases[mode]["success"],
                              "collisions": cases[mode].get("collision_steps"), "steps": metrics.get("control_intervals"),
                              "yaw_accel_max": metrics.get("angular_acceleration_radps2", {}).get("max"),
                              "in_place_fraction": metrics.get("rotating_in_place_fraction"),
                              "error": cases[mode].get("error", {}).get("message")}), flush=True)
        if all(mode in cases for mode in ["astar", "hybrid"]):
            old, new = cases["astar"].get("metrics", {}), cases["hybrid"].get("metrics", {})
            if old and new:
                entry["paired_differences"] = {
                    "yaw_acceleration_max_reduction_percent": 100 * (1 - new["angular_acceleration_radps2"]["max"] /
                        max(old["angular_acceleration_radps2"]["max"], 1e-12)),
                    "yaw_velocity_total_variation_reduction_percent": 100 * (1 - new["angular_velocity_total_variation_radps"] /
                        max(old["angular_velocity_total_variation_radps"], 1e-12)),
                    "in_place_fraction_change": new["rotating_in_place_fraction"] - old["rotating_in_place_fraction"],
                    "executed_path_length_change_m": new["path_length_m"] - old["path_length_m"],
                    "simulated_time_change_s": new["simulated_time_s"] - old["simulated_time_s"],
                }
        if not args.no_plots:
            plot_map(out, map_id, manifest, cases)
    report["phase"] = "complete"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["all_successful_zero_collisions"] = all(case["success"] and case.get("collision_steps") == 0
        for entry in report["maps"].values() for case in entry["cases"].values())
    atomic_json(destination, report)
    print(destination)
    raise SystemExit(0 if report["all_successful_zero_collisions"] else 1)


if __name__ == "__main__":
    main()
