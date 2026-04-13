# Sim2Real Runtime

This repository contains the runtime used to deploy the motion-tracking policy in:

- `sim2sim`: MuJoCo simulator + high-level controller
- `sim2real`: Unitree G1 + high-level controller

This environment is only for running the policy side.

Live teleoperation runs in a separate environment. See [`teleop/README.md`](./teleop/README.md).

## What This README Contains

This README is organized as follows:

1. `uv` setup for the policy/runtime environment
2. how to run `sim2sim`
3. how to run `sim2real`
4. what the UDP motion selector is and how to use it
5. what the VR motion source is and how to use it

## Setup With `uv`

Use `uv` for this repository. Do not create a separate Conda environment for the policy runtime.

```bash
cd /path/to/sim2real
uv sync
```

That creates the local `.venv` from `pyproject.toml` and `uv.lock`.

Run all scripts through `uv run`:

```bash
uv run src/sim2sim.py --xml_path assets/g1/g1.xml
uv run src/deploy.py --net lo --sim2sim
uv run src/motion_select.py
```

## Run `sim2sim`

`sim2sim` uses two processes:

1. the simulator / low-level state publisher
2. the high-level controller

Start the simulator first:

```bash
cd /path/to/sim2real
uv run src/sim2sim.py --xml_path assets/g1/g1.xml
```

Then start the controller in another terminal:

```bash
cd /path/to/sim2real
uv run src/deploy.py --net lo --sim2sim
```

Flow:

1. keep the simulator window focused so the simulated remote input works
2. press `s` in the simulator to leave zero-torque / move to default pose
3. once the robot is in the default pose, press `a` in the simulator to start the tracking policy
4. press `x` to exit

If your motion source is `udp`, also run the motion selector in a third terminal.

If your motion source is `vr`, also run the teleop bridge from [`teleop/README.md`](./teleop/README.md).

## Run `sim2real`

`sim2real` only starts the high-level controller in this repository. The robot hardware side is the real Unitree platform.

Before running:

1. power on G1
2. connect your PC to the robot over Ethernet
3. configure the correct network interface on your PC
4. make sure you know the interface name you want to pass as `--net`

Start the controller:

```bash
cd /path/to/sim2real
uv run src/deploy.py --net <robot_iface> --real
```

Flow:

1. controller starts in zero-torque mode and waits for the remote `start` button
2. press `start` on the Unitree remote to move the robot to the default pose
3. place or confirm the robot is safely on the ground
4. press `A` on the Unitree remote to enter the tracking policy
5. press `select` on the Unitree remote to exit

Always test a motion in `sim2sim` before running it on the real robot.

## Motion Sources

The tracking policy can consume two kinds of motion sources:

- `udp`
- `vr`

This is configured in [`config/tracking.yaml`](./config/tracking.yaml) with:

```yaml
motion_source: "vr"
```

The current default in this repository is `vr`.

## UDP Motion Selector

### What it is

The UDP motion selector is the offline motion-switching interface.

In this mode, the controller does not consume live teleop data. Instead, it plays motions listed in [`config/tracking.yaml`](./config/tracking.yaml), and you choose which motion to append through a small UDP command tool.

Internally:

- `deploy.py` creates a `UDPMotionSource`
- `UDPMotionSource` starts a tiny UDP server
- `motion_select.py` sends motion names to that UDP server

### How to use it

First change the tracking config:

```yaml
motion_source: "udp"
```

Then run the normal controller flow:

- `sim2sim`:
  ```bash
  uv run src/sim2sim.py --xml_path assets/g1/g1.xml
  uv run src/deploy.py --net lo --sim2sim
  ```
- `sim2real`:
  ```bash
  uv run src/deploy.py --net <robot_iface> --real
  ```

Then run the selector in another terminal:

```bash
cd /path/to/sim2real
uv run src/motion_select.py
```

Usage:

- type a motion index or motion name and press Enter
- type `list` to print all available motions
- press Enter on an empty line to resend the previous choice
- type `r` to reload `config/tracking.yaml`
- type `q` to quit

Behavior:

- `default` returns the policy toward the idle/default pose
- non-default motions are taken from the `motions:` list in `config/tracking.yaml`
- switching is append-based rather than an immediate hard cut
- switching follows the policy-side gating logic:
  - from `default`, you can switch to any motion
  - once a non-default motion is active, you cannot jump directly to another non-default motion
      - a non-default motion must finish first
      - after it finishes, you can switch back to `default`
      - only after returning to `default` can you switch to a different motion

## VR Motion Source

### What it is

The VR motion source is the live teleoperation interface.

In this mode, `sim2sim/sim2real` does not receive motion names over UDP. Instead, it requests pose chunks from the teleop bridge over ZMQ.

Internally:

- `deploy.py` creates a `VRMotionSource`
- `VRMotionSource` connects to the teleop bridge on the ZMQ addresses in [`config/tracking.yaml`](./config/tracking.yaml)
- `VRMotionSource` maintains the reference-motion buffer
- when the future horizon drops below the low-water mark, it requests more frames
- the teleop bridge retargets the latest XR/PICO stream and returns a chunk of frames

The teleop bridge itself is documented in [`teleop/README.md`](./teleop/README.md).

### How to use it

Leave the config as:

```yaml
motion_source: "vr"
```

Start the teleop bridge in its own environment by following [`teleop/README.md`](./teleop/README.md).

Then start the controller here as usual:

- `sim2sim`:
  ```bash
  uv run src/sim2sim.py --xml_path assets/g1/g1.xml
  uv run src/deploy.py --net lo --sim2sim
  ```
- `sim2real`:
  ```bash
  uv run src/deploy.py --net <robot_iface> --real
  ```

### Buttons

There are two layers of control in VR mode.

Robot-side controller state:

- simulated remote in `sim2sim`: `s` to move to default pose, `a` to start tracking
- Unitree remote in `sim2real`: `start` to move to default pose, `A` to start tracking

Live teleop control from the XR/PICO side:

- right-hand `A`: start/resume live teleop streaming
- left-hand `X`: pause live teleop streaming

The XR/PICO installation and teleop-side runtime are intentionally kept out of this `uv` environment. Use the separate setup in [`teleop/README.md`](./teleop/README.md).
