"""Language-conditioned visual imitation demo for a fixed-base Unitree G1."""
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np


COLORS = ("azul", "marrom")
ENGLISH_COLORS = {"azul": "blue", "marrom": "brown"}
INPUT_Z = 0.94
OUTPUT_Z = 0.92
HALF_BOX = np.array([0.055, 0.043, 0.04])
DESTINATION = np.array([0.34, 0.0, OUTPUT_Z + HALF_BOX[2] + 0.003])
HAND_OFFSET = np.array([0.0, 0.0, 0.065])


def add(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {
        key: value if isinstance(value, str) else " ".join(map(str, np.atleast_1d(value)))
        for key, value in attributes.items()
    })


def build(assets):
    root = ET.parse(assets / "g1_with_hands.xml").getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((assets / compiler.get("meshdir", "assets")).resolve()))
    root.find("option").set("timestep", "0.005")
    world = root.find("worldbody")
    pelvis = world.find("./body[@name='pelvis']")
    pelvis.remove(pelvis.find("freejoint"))
    root.remove(root.find("keyframe"))
    # The fixed-base demonstration uses collision layers for the box and worktops.
    for geom in pelvis.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    for side in ("left", "right"):
        wrist = pelvis.find(f".//body[@name='{side}_wrist_yaw_link']")
        add(wrist, "site", name=f"hand_{side}", pos=[0.05, 0, 0], size=0.009)
    add(world, "light", pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.9] * 3)
    add(world, "camera", name="producao", pos=[0.36, 0, 2.2],
        xyaxes=[0, 1, 0, -1, 0, 0], fovy=40)
    add(world, "geom", type="plane", size=[3, 3, 0.1], rgba=[0.22, 0.25, 0.28, 1])
    for y, z, color, width in [
        (-0.28, INPUT_Z, [0.34, 0.37, 0.4, 1], 0.105),
        (0.28, INPUT_Z, [0.34, 0.37, 0.4, 1], 0.105),
        (0, OUTPUT_Z, [0.21, 0.39, 0.27, 1], 0.11),
    ]:
        add(world, "geom", type="box", pos=[0.35, y, z - 0.025],
            size=[0.14, width, 0.025], rgba=color)
        for x in (0.24, 0.46):
            add(world, "geom", type="box", pos=[x, y, z / 2 - 0.025],
                size=[0.016, 0.035, z / 2 - 0.025], rgba=[0.31, 0.34, 0.37, 1])
    for color, rgba in zip(COLORS, ([0.03, 0.23, 0.95, 1], [0.73, 0.31, 0.10, 1])):
        body = add(world, "body", name=color, pos=[0.35, 0, INPUT_Z + HALF_BOX[2]])
        add(body, "freejoint", name=f"caixa_{color}")
        add(body, "geom", type="box", size=HALF_BOX, mass=0.6, rgba=rgba)
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def boxes(model):
    return {color: model.joint(f"caixa_{color}").qposadr.item() for color in COLORS}


def reset(model, data, rng):
    mj.mj_resetData(model, data)
    for (color, address), side in zip(boxes(model).items(), rng.permutation([-1, 1])):
        data.qpos[address:address + 3] = [
            rng.uniform(0.31, 0.39), side * rng.uniform(0.26, 0.30), INPUT_Z + HALF_BOX[2],
        ]
    mj.mj_forward(model, data)


def image(renderer, data):
    renderer.update_scene(data, camera="producao")
    return renderer.render()


def observation(rgb, color):
    red, green, blue = rgb.astype(float).transpose(2, 0, 1)
    if color == "azul":
        mask = (blue > 105) & (blue > 1.55 * red) & (blue > 1.25 * green)
    else:
        mask = (red > 105) & (red > 1.28 * green) & (red > 1.55 * blue)
    rows, columns = np.nonzero(mask)
    if len(rows) < 15:
        raise RuntimeError(f"The camera could not find the {ENGLISH_COLORS[color]} box.")
    return np.array([1, columns.mean() / rgb.shape[1], rows.mean() / rgb.shape[0]])


def train(model, data, renderer, samples, rng, weights, dataset, viewer):
    features, targets, images, labels = [], [], [], []
    for _ in range(samples):
        if not viewer.is_running():
            return None
        reset(model, data, rng)
        viewer.sync()
        rgb = image(renderer, data)
        time.sleep(0.08)
        for color, address in boxes(model).items():
            features.append(observation(rgb, color))
            targets.append(data.qpos[address:address + 3].copy())
            if dataset:
                images.append(rgb.copy())
                labels.append(f"Pick up the {ENGLISH_COLORS[color]} box and place it on the output bench.")
    policy = np.linalg.lstsq(features, targets, rcond=None)[0]
    weights.parent.mkdir(parents=True, exist_ok=True)
    np.save(weights, policy)
    if dataset:
        np.savez_compressed(dataset, rgb=images, instruction=labels,
                            features=features, grasp_xyz=targets)
    return policy


def arm(model, side):
    names = ["waist_yaw_joint"] + [f"{side}_{joint}_joint" for joint in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw",
    )]
    return [model.joint(name).id for name in names]


def finger_pose(model, side):
    """Return the joint addresses and mirrored flexion targets for the G1 hand."""
    names = ("thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
    flexion = np.array([0.38, 0.82, -0.88, -0.93, -0.88, -0.93])
    if side == "right":
        flexion = -flexion
    joints = [model.joint(f"{side}_hand_{name}_joint").id for name in names]
    return model.jnt_qposadr[joints], flexion


def approach(model, data, site, joints, target):
    jacobian = np.zeros((3, model.nv))
    dofs = model.jnt_dofadr[joints]
    positions = model.jnt_qposadr[joints]
    for _ in range(10):
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        error = target - data.site_xpos[site]
        if np.linalg.norm(error) < 0.002:
            break
        mj.mj_jacSite(model, data, jacobian, None, site)
        jac = jacobian[:, dofs]
        delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.002 * np.eye(3), error)
        limits = model.jnt_range[joints]
        data.qpos[positions] = np.clip(
            data.qpos[positions] + np.clip(delta, -0.085, 0.085), limits[:, 0], limits[:, 1],
        )
    mj.mj_kinematics(model, data)


class Physics:
    def __init__(self, model, data, device):
        self.model, self.data = model, data
        self.gpu = device == "cuda:0"
        if self.gpu:
            import mujoco_warp as mjw
            import warp as wp
            wp.init()
            self.wp = wp
            with wp.ScopedDevice(device):
                self.model_gpu = mjw.put_model(model)
                self.data_gpu = mjw.put_data(model, data)
                mjw.step(self.model_gpu, self.data_gpu)
                wp.synchronize()
                with wp.ScopedCapture() as capture:
                    for _ in range(6):
                        mjw.step(self.model_gpu, self.data_gpu)
                self.graph = capture.graph
            print(f"Physics: {self.data_gpu.qpos.device}")
        else:
            print("Physics: CPU")

    def step(self):
        m, d = self.model, self.data
        robot_size = m.nq - len(COLORS) * 7
        robot_pose = d.qpos[:robot_size].copy()
        if self.gpu:
            with self.wp.ScopedDevice("cuda:0"):
                self.data_gpu.qpos.assign(d.qpos[None, :].astype(np.float32))
                self.data_gpu.qvel.assign(d.qvel[None, :].astype(np.float32))
                self.wp.capture_launch(self.graph)
                d.qpos[:] = self.data_gpu.qpos.numpy()[0]
                d.qvel[:] = self.data_gpu.qvel.numpy()[0]
                d.time += 6 * m.opt.timestep
        else:
            mj.mj_step(m, d, nstep=6)
        d.qpos[:robot_size] = robot_pose
        d.qvel[:robot_size] = 0


def run(model, data, renderer, policy, physics, color, viewer, record=None):
    grasp = observation(image(renderer, data), color) @ policy
    side = "left" if grasp[1] > 0 else "right"
    site = model.site(f"hand_{side}").id
    joints = arm(model, side)
    address = boxes(model)[color]
    velocity = model.joint(f"caixa_{color}").dofadr.item()
    print(f"Visual target: {ENGLISH_COLORS[color]} at {grasp.round(3)}; hand: {side}")
    pickup = grasp + HAND_OFFSET
    dropoff = DESTINATION + HAND_OFFSET
    clearance = np.array([0, 0, 0.065])
    # Separate the slow approach and placement from the faster transit.
    waypoints = [
        (pickup + clearance, False, 25),
        (pickup, False, 32),
        (pickup, True, 10),
        (pickup + clearance, True, 32),
        (dropoff + clearance, True, 42),
        (dropoff, True, 38),
        (dropoff, True, 10),
        (dropoff, False, 12),
        (dropoff + clearance, False, 26),
    ]
    fingers, closed_angles = finger_pose(model, side)
    grip = 0.0
    for index, (goal, carrying, frames) in enumerate(waypoints):
        start_grip = grip
        start = data.site_xpos[site].copy()
        for fraction in np.linspace(1 / frames, 1, frames):
            if viewer and not viewer.is_running():
                return None
            begin = time.monotonic()
            blend = fraction**3 * (10 + fraction * (-15 + 6 * fraction))
            if record is not None:
                record.observe(model, data, renderer)
            approach(model, data, site, joints, start + blend * (goal - start))
            grip = start_grip + blend * (float(carrying) - start_grip)
            data.qpos[fingers] = grip * closed_angles
            if record is not None:
                record.action(data, side, grip)
            if grip >= 0.65:
                data.qpos[address:address + 3] = data.site_xpos[site] - HAND_OFFSET
                data.qvel[velocity:velocity + 6] = 0
            physics.step()
            if grip >= 0.65:
                data.qpos[address:address + 3] = data.site_xpos[site] - HAND_OFFSET
                data.qvel[velocity:velocity + 6] = 0
            mj.mj_forward(model, data)
            if viewer:
                viewer.sync()
                time.sleep(max(0.0, 0.03 - (time.monotonic() - begin)))
        if index == 1 and np.linalg.norm(data.site_xpos[site] - (data.qpos[address:address + 3] + HAND_OFFSET)) > 0.045:
            print("Failure: the hand did not reach the box.")
            return False

    selected = data.qpos[address:address + 3]
    other = next(a for c, a in boxes(model).items() if c != color)
    success = bool(np.linalg.norm(selected[:2] - DESTINATION[:2]) < 0.065
                   and abs(selected[2] - OUTPUT_Z - HALF_BOX[2]) < 0.025
                   and abs(data.qpos[other + 1]) > 0.18)
    print(f"{'SUCCESS' if success else 'FAILURE'}: {ENGLISH_COLORS[color]} box at {selected.round(3)}")
    return success
