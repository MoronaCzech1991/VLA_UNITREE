"""Launch the official Isaac Lab Arena G1 whole-body GR00T demo with a GUI."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", required=True, type=Path, help="Isaac Lab Arena checkout")
    parser.add_argument("--host", default="127.0.0.1", help="GR00T policy server host")
    parser.add_argument("--port", type=int, default=5555, help="GR00T policy server port")
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.port <= 65535:
        parser.error("Use a positive step count and a valid TCP port.")
    arena = args.arena.resolve()
    runner = "isaaclab_arena/evaluation/policy_runner.py"
    config = "isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml"
    for path in (runner, config):
        if not (arena / path).is_file():
            parser.error(f"Missing {arena / path}. Use the compatible Isaac Lab Arena checkout.")
    return subprocess.run([
        sys.executable, runner, "--viz", "kit",
        "--policy_type", "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy",
        "--policy_config_yaml_path", config,
        "--remote_host", args.host, "--remote_port", str(args.port),
        "--num_steps", str(args.steps), "--enable_cameras",
        "galileo_g1_locomanip_pick_and_place", "--object", "brown_box",
        "--embodiment", "g1_wbc_joint",
    ], cwd=arena, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
