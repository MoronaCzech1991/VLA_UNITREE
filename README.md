# Unitree G1 | SmolVLA factory logistics

The Unitree G1 transfers a blue or brown box to an output bench in a simulated industrial station. Every simulation opens the MuJoCo viewer. Both colors move between starting positions across episodes.

## Who controls the robot?

In `run` mode, a trained **SmolVLA checkpoint** receives the RGB camera image, robot joint state, and complete text instruction. Its action commands set 15 G1 joint targets and the two simulated gripper commands. The simulator applies limits, checks whether the commanded gripper is near a box, and simulates the release. The model chooses which box to approach, which hand to use, and when to close and open the gripper.

`collect` mode uses a scripted teacher solely to record demonstrations for training. The teacher does not run during policy inference. The G1 has a fixed base and an assisted grasp, suitable for testing box selection and transfer in simulation. A trained G1 checkpoint is required and is not included.

## Install in Ubuntu-MuJoCo

```bash
cd ~/VLA_UNITREE
git pull --ff-only origin main
source ~/mujoco-demo/.venv/bin/activate
uv pip install "lerobot[smolvla]"
unset LIBGL_ALWAYS_SOFTWARE
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

For a new setup, clone this repository and install `requirements.txt` into your environment. Get the robot assets from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1). The examples below assume the assets are already present in your original MuJoCo directory.

## 1. Collect visible training demonstrations

```bash
python demo.py collect \
  --assets ~/mujoco-demo/mujoco_menagerie/unitree_g1 \
  --device cuda:0 --episodes 60 --dataset runs/g1-boxes
```

Collection alternates blue and brown instructions. The visible teacher saves successful episodes as a local LeRobot dataset. Each frame records camera RGB, robot state, the complete task instruction, and joint and gripper actions. Close the viewer to stop collection early and save completed episodes.

## 2. Fine-tune SmolVLA on the G1 action space

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features='{"observation.images.camera1":{"type":"VISUAL","shape":[3,240,320]},"observation.state":{"type":"STATE","shape":[17]}}' \
  --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=local/g1-boxes --dataset.root=runs/g1-boxes \
  --batch_size=1 --steps=10000 --output_dir=runs/g1-smolvla
```

This is a starting configuration. Training and inference memory requirements need to be checked on the target GPU. If training exceeds available VRAM, train on a larger GPU and copy the resulting checkpoint to the PC. See the [SmolVLA fine-tuning guide](https://huggingface.co/docs/lerobot/main/smolvla).

## 3. Watch SmolVLA control the robot

```bash
python demo.py run \
  --assets ~/mujoco-demo/mujoco_menagerie/unitree_g1 \
  --checkpoint runs/g1-smolvla/checkpoints/last/pretrained_model \
  --device cuda:0 --policy-device cuda --seed 73 \
  --instruction "Pick up the blue box and place it on the output bench."
```

Change `blue` to `brown` to test the other condition. Use seeds distinct from collection for evaluation. The terminal reports which box reached the output after the episode. The simulation remains visible until you close the window.
