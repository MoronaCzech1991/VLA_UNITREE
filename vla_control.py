"""Collect visible demonstrations or run a trained SmolVLA policy on the G1."""
import argparse
from pathlib import Path
import time

import mujoco as mj
from mujoco import viewer as native_viewer
import numpy as np

from teacher import (
    COLORS, DESTINATION, ENGLISH_COLORS, HALF_BOX, HAND_OFFSET, OUTPUT_Z,
    Physics, arm, boxes, build, finger_pose, image, reset, run, train,
)


CAMERA = "observation.images.camera1"
STATE = "observation.state"
ACTION = "action"
FPS = 33  # Six 5 ms physics steps per action, rounded for LeRobot timestamps.
SIDES = ("left", "right")


class Robot:
    """Absolute targets for 15 G1 joints plus two simulated gripper commands."""
    def __init__(self, model):
        self.model = model
        self.joints = np.array(arm(model, "left") + arm(model, "right")[1:])
        self.addresses = model.jnt_qposadr[self.joints]
        self.limits = model.jnt_range[self.joints]
        self.names = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, index)
            for index in self.joints
        ] + ["left_grip", "right_grip"]
        self.boxes = boxes(model)
        self.box_velocities = {
            color: model.joint(f"caixa_{color}").dofadr.item() for color in COLORS
        }
        self.sites = [model.site(f"hand_{side}").id for side in SIDES]
        self.held = [None, None]
        self.grip_request = np.zeros(2, dtype=bool)
        self.grip_position = np.zeros(2, dtype=np.float32)
        self.fingers = [finger_pose(model, side) for side in SIDES]

    def state(self, data):
        return np.array([*data.qpos[self.addresses], *self.grip_position],
                        dtype=np.float32)

    def apply(self, data, action, physics):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (len(self.names),) or not np.isfinite(action).all():
            raise ValueError(f"Expected {len(self.names)} finite G1 action values, got {action.shape}.")
        current = data.qpos[self.addresses]
        desired = np.clip(action[:-2], self.limits[:, 0], self.limits[:, 1])
        data.qpos[self.addresses] = current + np.clip(desired - current, -0.10, 0.10)
        desired_grip = np.clip(action[-2:], 0.0, 1.0)
        self.grip_position += np.clip(desired_grip - self.grip_position, -0.12, 0.12)
        for grip, (addresses, angles) in zip(self.grip_position, self.fingers):
            data.qpos[addresses] = grip * angles
        mj.mj_kinematics(self.model, data)

        # Debounce marginal predictions so a noisy action cannot chatter the gripper.
        self.grip_request[self.grip_position >= 0.65] = True
        self.grip_request[self.grip_position <= 0.35] = False
        for hand, (site, grip) in enumerate(zip(self.sites, self.grip_request)):
            if not grip:
                self.held[hand] = None
            elif self.held[hand] is None:
                target = data.site_xpos[site] - HAND_OFFSET
                available = [color for color in COLORS if color not in self.held]
                nearest = min(available, key=lambda color: np.linalg.norm(
                    data.qpos[self.boxes[color]:self.boxes[color] + 3] - target
                ), default=None)
                if nearest is not None:
                    address = self.boxes[nearest]
                    if np.linalg.norm(data.qpos[address:address + 3] - target) < 0.055:
                        self.held[hand] = nearest

        self.update_grasps(data)
        physics.step()
        self.update_grasps(data)
        mj.mj_forward(self.model, data)

    def update_grasps(self, data):
        for site, color in zip(self.sites, self.held):
            if color is not None:
                address = self.boxes[color]
                velocity = self.box_velocities[color]
                data.qpos[address:address + 3] = data.site_xpos[site] - HAND_OFFSET
                data.qvel[velocity:velocity + 6] = 0


class Recorder:
    def __init__(self, robot, instruction):
        self.robot = robot
        self.instruction = instruction
        self.grip = np.zeros(2, dtype=np.float32)
        self.frames = []

    def observe(self, model, data, renderer):
        state = np.concatenate((data.qpos[self.robot.addresses], self.grip)).astype(np.float32)
        self.frames.append({CAMERA: image(renderer, data).copy(), STATE: state,
                            "task": self.instruction})

    def action(self, data, side, amount):
        self.grip = np.array([amount if side == hand else 0 for hand in SIDES], dtype=np.float32)
        self.frames[-1][ACTION] = np.concatenate((
            data.qpos[self.robot.addresses], self.grip
        )).astype(np.float32)


class SmolVLA:
    def __init__(self, checkpoint, device, robot):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        config = self.policy.config
        expected = {STATE: (len(robot.names),), CAMERA: (3, 240, 320)}
        for key, shape in expected.items():
            feature = config.input_features.get(key)
            if feature is None or tuple(feature.shape) != shape:
                raise ValueError(f"Checkpoint needs {key} with shape {shape}. Train on the G1 dataset.")
        if ACTION not in config.output_features or tuple(config.output_features[ACTION].shape) != (len(robot.names),):
            raise ValueError("Checkpoint action dimensions do not match the G1 actuator interface.")
        self.preprocess, self.postprocess = make_pre_post_processors(
            config, checkpoint, preprocessor_overrides={"device_processor": {"device": device}},
        )

    def action(self, rgb, state, instruction):
        torch = self.torch
        observation = {
            CAMERA: torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0,
            STATE: torch.from_numpy(state),
            "task": instruction,
        }
        with torch.inference_mode():
            predicted = self.policy.select_action(self.preprocess(observation))
            action = self.postprocess(predicted).detach().cpu().numpy()
            return action[0] if action.ndim == 2 and action.shape[0] == 1 else action


def collect(args, model, data, renderer, robot, physics, viewer, rng):
    from lerobot.datasets import LeRobotDataset

    if args.dataset.exists():
        raise FileExistsError(f"Dataset already exists: {args.dataset}. Choose a new path.")
    policy = train(model, data, renderer, 24, np.random.default_rng(810),
                   Path("runs/teacher.npy"), None, viewer)
    if policy is None:
        return
    features = {
        CAMERA: {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channel"]},
        STATE: {"dtype": "float32", "shape": (len(robot.names),), "names": robot.names},
        ACTION: {"dtype": "float32", "shape": (len(robot.names),), "names": robot.names},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/g1-boxes", root=args.dataset, fps=FPS,
        robot_type="unitree_g1_fixed", features=features, use_videos=False,
    )
    saved = 0
    try:
        for episode in range(args.episodes):
            if not viewer.is_running():
                break
            reset(model, data, rng)
            color = COLORS[episode % len(COLORS)]
            instruction = f"Pick up the {ENGLISH_COLORS[color]} box and place it on the output bench."
            record = Recorder(robot, instruction)
            if run(model, data, renderer, policy, physics, color, viewer, record):
                for frame in record.frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                saved += 1
                print(f"Saved demonstration {saved}/{args.episodes}.", flush=True)
    finally:
        dataset.finalize()
    if not saved:
        raise RuntimeError("No successful demonstrations saved.")


def rollout(args, model, data, renderer, robot, physics, viewer, rng):
    policy = SmolVLA(args.checkpoint, args.policy_device, robot)
    for episode in range(args.episodes):
        if not viewer.is_running():
            break
        reset(model, data, rng)
        robot.held[:] = [None, None]
        robot.grip_request[:] = False
        robot.grip_position[:] = 0
        policy.policy.reset()
        print(f"SmolVLA episode {episode + 1}: {args.instruction}", flush=True)
        for _ in range(args.steps):
            if not viewer.is_running():
                return
            start = time.monotonic()
            action = policy.action(image(renderer, data), robot.state(data), args.instruction)
            robot.apply(data, action, physics)
            viewer.sync()
            time.sleep(max(0, 1 / FPS - (time.monotonic() - start)))
        for color, address in robot.boxes.items():
            position = data.qpos[address:address + 3]
            placed = (color not in robot.held
                      and np.linalg.norm(position[:2] - DESTINATION[:2]) < 0.065
                      and abs(position[2] - OUTPUT_Z - HALF_BOX[2]) < 0.025)
            print(f"{ENGLISH_COLORS[color]} box on output bench: {placed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "run"])
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("runs/g1-boxes"))
    parser.add_argument("--checkpoint", help="Path or repo ID of a fine-tuned G1 SmolVLA checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu", help="Physics device")
    parser.add_argument("--policy-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", default="Pick up the blue box and place it on the output bench.")
    args = parser.parse_args()
    if args.mode == "run" and not args.checkpoint:
        parser.error("run requires --checkpoint with a trained G1 SmolVLA policy.")
    if args.steps < 1 or args.episodes < 1:
        parser.error("--steps and --episodes must be positive.")
    if not (args.assets / "g1_with_hands.xml").is_file():
        parser.error("--assets must point to the unitree_g1 directory containing g1_with_hands.xml.")
    model = build(args.assets)
    data = mj.MjData(model)
    rng = np.random.default_rng(args.seed)
    reset(model, data, rng)
    robot = Robot(model)
    with native_viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.19, 0, 0.79]
            viewer.cam.distance = 2.35
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -24
        viewer.sync()
        with mj.Renderer(model, height=240, width=320) as renderer:
            physics = Physics(model, data, args.device)
            execute = collect if args.mode == "collect" else rollout
            execute(args, model, data, renderer, robot, physics, viewer, rng)
            print("Close the viewer to exit.", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)


if __name__ == "__main__":
    main()
