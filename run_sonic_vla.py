"""Run the pretrained Unitree G1 pi0.5 policy in MuJoCo with a live Rerun view."""
import argparse
import json
import os
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Policy inference device; CPU avoids allocating model weights in VRAM")
    args = parser.parse_args()
    rollout = shutil.which("lerobot-rollout")
    if rollout is None:
        parser.error("Activate the LeRobot G1 environment before launching this demo.")

    cameras = {"ego_view": {
        "type": "zmq", "server_address": "127.0.0.1", "port": 5555,
        "camera_name": "head_camera", "width": 640, "height": 480,
        "fps": 30, "warmup_s": 15,
    }}
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    print(f"Model: nepyope/sonic_walk (pi0.5); inference: {args.device}; viewer: Rerun",
          flush=True)
    return subprocess.run([
        rollout,
        "--strategy.type=base",
        "--policy.path=nepyope/sonic_walk",
        f"--policy.device={args.device}",
        "--robot.type=unitree_g1",
        "--robot.is_simulation=true",
        "--robot.controller=SonicWholeBodyController",
        f"--robot.cameras={json.dumps(cameras)}",
        "--task=walk forward",
        "--display_data=true",
    ], env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
