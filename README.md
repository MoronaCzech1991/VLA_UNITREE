# Unitree G1 VLA | Factory logistics demo

The Unitree G1 receives an instruction, locates a blue or brown box using a simulated RGB camera, and places the selected box on the output bench. Box positions vary between episodes. The factory workstation is a generic simulation scene.

**The MuJoCo viewer opens on every run.** Watch the boxes change positions during calibration, then watch the robot place the selected box. The window stays open after the last episode. Close it to exit.

## Run on Ubuntu-MuJoCo

Reuse the Python environment and Unitree G1 assets from the earlier MuJoCo setup:

```bash
cd ~
git clone https://github.com/MoronaCzech1991/VLA_UNITREE.git
cd VLA_UNITREE
source ~/mujoco-demo/.venv/bin/activate

env -u LIBGL_ALWAYS_SOFTWARE \
  GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  python demo.py --assets ~/mujoco-demo/mujoco_menagerie/unitree_g1 \
  --device cuda:0 \
  --instruction "Pick up the blue box and place it on the output bench"
```

If you already cloned the repository, run `git pull --ff-only origin main` inside `~/VLA_UNITREE` first. Replace `blue` with `brown` to pick the other box. The words `azul` and `marrom` also work. Use `--seed 43` to change the starting layout and `--episodes 3` to watch three attempts in one window.

The CUDA mode uses MuJoCo Warp for physics and runs inverse kinematics and visual target estimation on the CPU. On WSL, D3D12 handles the display. On a graphical Linux desktop, run the same Python command without the preceding `env` lines. Pass `--device cpu` to use the CPU for physics.

## Standalone installation

Requires Python 3.12 or later, Git, and a graphical session. In a fresh clone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git assets
git -C assets sparse-checkout set unitree_g1
```

In this setup, use `--assets assets/unitree_g1` in the run command. The [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) provides the original robot model and meshes.

## Watch calibration and evaluation

The first launch calibrates the visual policy automatically. Pass `--train` to recalibrate and `--dataset runs/calibration.npz` to save the examples. For example, with the WSL display environment shown above:

```bash
env -u LIBGL_ALWAYS_SOFTWARE \
  GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
  python demo.py --assets ~/mujoco-demo/mujoco_menagerie/unitree_g1 \
  --device cuda:0 --train --dataset runs/calibration.npz --episodes 3 \
  --instruction "Pick up the brown box"
```

The terminal reports an episode as successful when the selected box is supported by the output bench and the other box remains at the input.

## How the prototype works

The instruction parser selects a color. RGB segmentation identifies the corresponding box, and a regression trained on camera examples estimates its 3D position. The controller executes a fixed pick-and-place sequence using inverse kinematics. The G1 has a fixed base and an assisted grasp. MuJoCo simulates gravity and contact after release. The scene is intended for demonstrating the task flow. Grasp force, whole-body balance, and locomotion are outside its scope.

The prototype uses a modular vision-language-action policy. SmolVLA integration and training are pending. The optional NPZ stores camera images, instructions, visual features, and approach targets. Adapting SmolVLA requires recording time-aligned observations and actions, converting those episodes to LeRobot format, and training or adapting a checkpoint for the G1 action space.
