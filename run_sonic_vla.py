"""Run the pretrained Unitree G1 pi0.5 policy in MuJoCo with a live Rerun view."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from urllib.parse import urlencode


def wait_for_viewer(process, host, ports):
    for port in ports:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError(f"Rerun exited with code {process.returncode}.")
            try:
                with socket.create_connection((host, port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError(f"Rerun did not start on {host}:{port}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                        help="Policy inference device; CPU avoids allocating model weights in VRAM")
    parser.add_argument("--checkpoint", type=Path,
                        help="Local LeRobot pi0.5 checkpoint directory for the G1 SONIC controller")
    parser.add_argument("--viewer", choices=("web", "native"), default="web")
    parser.add_argument("--viewer-port", type=int, default=9090)
    parser.add_argument("--stream-port", type=int, default=9876)
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.viewer_port, args.stream_port)):
        parser.error("Viewer ports must be between 1 and 65535.")
    if args.viewer_port == args.stream_port:
        parser.error("The web and stream ports must be different.")
    if args.checkpoint is None:
        parser.error("--checkpoint is required. See the README for checkpoint setup and GPU requirements.")
    checkpoint = args.checkpoint.expanduser().resolve()
    for filename in ("config.json", "model.safetensors"):
        if not (checkpoint / filename).is_file():
            parser.error(f"Missing checkpoint file: {checkpoint / filename}")
    try:
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        parser.error(f"Cannot read checkpoint configuration: {error}")
    if (config.get("type") != "pi05"
            or config.get("input_features", {}).get("observation.images.ego_view", {}).get("shape") != [3, 480, 640]
            or config.get("input_features", {}).get("observation.state", {}).get("shape") != [64]
            or config.get("output_features", {}).get("action", {}).get("shape") != [64]):
        parser.error("Expected a pi0.5 SONIC checkpoint with ego_view RGB and 64-value state and actions.")
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
    print(f"Model: {checkpoint} (pi0.5); inference: {args.device}; viewer: Rerun {args.viewer}",
          flush=True)
    command = [
        rollout,
        "--strategy.type=base",
        f"--policy.path={checkpoint}",
        f"--policy.device={args.device}",
        "--robot.type=unitree_g1",
        "--robot.is_simulation=true",
        "--robot.controller=SonicWholeBodyController",
        f"--robot.cameras={json.dumps(cameras)}",
        "--task=walk forward",
        "--display_data=true",
    ]
    if args.viewer == "native":
        return subprocess.run(command, env=environment, check=False).returncode

    rerun = shutil.which("rerun")
    if rerun is None:
        parser.error("Install the viewer: uv pip install 'lerobot[viz]==0.7.0'.")
    host = "127.0.0.1"
    process = subprocess.Popen([
        rerun, "--serve-web", "--bind", host,
        "--port", str(args.stream_port), "--web-viewer-port", str(args.viewer_port),
    ], env=environment)
    try:
        wait_for_viewer(process, host, (args.stream_port, args.viewer_port))
        stream = f"rerun+http://{host}:{args.stream_port}/proxy"
        print(f"Open the live viewer in Windows: http://{host}:{args.viewer_port}/?{urlencode({'url': stream, 'renderer': 'webgl'})}", flush=True)
        return subprocess.run([
            *command, f"--display_ip={host}", f"--display_port={args.stream_port}",
        ], env=environment, check=False).returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
