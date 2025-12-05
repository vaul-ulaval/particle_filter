# Particle Filter Localization

This code implements the MCL algorithm for the RACECAR.

[![YouTube Demo](./media/thumb.jpg)](https://www.youtube.com/watch?v=-c_0hSjgLYw)

For high efficiency in Python, it uses Numpy arrays and [RangeLibc](https://github.com/f1tenth/range_libc) for fast 2D ray casting.

# Installation

To run this, you need to ensure that both the map_server ROS package, and the python wrappers for RangeLibc are installed.

For the map server :

```
# Do this at the root of your colcon workspace
sudo apt-get update
rosdep install -r --from-paths src --ignore-src --rosdistro foxy -y
```

For [RangeLibc](https://github.com/f1tenth/range_libc):

```
# No need to clone this in your workspace, we will only use the python wrapper
git clone https://github.com/f1tenth/range_libc
sudo apt-get install python3-dev cython3
cd range_libc/pywrappers
# on VM
./compile.sh
# on car - compiles GPU ray casting methods
./compile_with_cuda.sh
```

# Utilisation dans le simulateur

Pour l'utiliser dans le simulateur, il faut commenter le lancement du map_server dans le launch file. Dans le fichier `launch/localize_launch.py`, commenter ces deux lignes (Vers la fin du fichier) :

```python
    ld.add_action(nav_lifecycle_node)
    ld.add_action(map_server_node)
```

Ensuite, dans le fichier `config/localize.yaml`, change le 'odometry_topic' pour '/ego_racecar/odom' et 'range_method' pour 'glt'

# Usage

The majority of parameters you might want to tweak are in the launch/localize.launch file. You may have to modify the "odometry_topic" or "scan_topic" parameters to match your environment.

```
ros2 launch particle_filter localize_launch.py
```

Once the particle filter is running, you can visualize the map and other particle filter visualization message in RViz. Use the "2D Pose Estimate" tool from the RViz toolbar to initialize the particle locations.

See [launch/localize.launch](/particle_filter/launch/localize_launch.py) for docs on available parameters and arguments.

The "range_method" parameter determines which RangeLibc ray casting method to use. The default is cddt because it is fast and has a low initialization time. The fastest option on the CPU is "glt" but it has a slow startup. The fastest version if you have can compile RangeLibc with CUDA enabled is "rmgpu". See this performance comparison chart:

![Range Method Performance Comparison](./media/comparison.png)

## Slip-aware motion model (straight-line launches)

The node now fuses two odometry sources to remain stable when the drive wheels spin faster than the chassis actually moves:

- `/odometry/filtered` (EKF output) supplies the pose delta applied to every particle.
- `/autodrive/roboracer_1/wheel_odom` (wheel-encoder speed) gives the instantaneous wheel speed used to detect slip.

Each time an odom message arrives the filter compares the EKF speed to the wheel speed and computes a slip ratio. When slip increases, two compensations kick in:

1. **Adaptive noise** – the Gaussian dispersions in `motionModel()` are scaled by `slip_scale_*` (with caps `slip_max_*`). This spreads particles so LiDAR can re-anchor them after the wheels stop lying.
2. **Odom attenuation** – the forward odom delta is multiplied by `attenuation = clamp(1 / (1 + slip_odom_gain * slip), slip_odom_min_factor, 1)`, meaning we trust encoder deltas less while slip persists.

### Tuning cheat sheet

| Parameter                     | Default                             | Effect                                                                                 |
| ----------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------- |
| `motion_dispersion_x/y/theta` | `0.04 / 0.025 / 0.25`               | Baseline odom noise (m, m, rad). Requires node restart after editing configs.          |
| `slip_alpha`                  | `0.2`                               | Low-pass factor for the slip ratio (0=slow, 1=immediate).                              |
| `slip_scale_{x,y,theta}`      | `3.0 / 0.4 / 1.0`                   | How aggressively each axis’ noise grows with slip.                                     |
| `slip_max_{x,y,theta}`        | `7 / 2 / 3`                         | Clamp to keep noise bounded.                                                           |
| `slip_odom_gain`              | `2.0`                               | Strength of odom attenuation. Larger values shrink odom deltas more when slip spikes.  |
| `slip_odom_min_factor`        | `0.2`                               | Minimum percentage of odom delta we will ever apply (prevents grinding to a halt).     |
| `slip_reference_floor`        | `0.5`                               | Minimum denominator for the slip ratio—prevents divide-by-zero when EKF speed is tiny. |
| `slip_wheel_timeout`          | `0.3 s`                             | How long to keep trusting the wheel-speed message before falling back to odom deltas.  |
| `wheel_speed_topic`           | `/autodrive/roboracer_1/wheel_odom` | Remap if your driver publishes wheel odom elsewhere.                                   |

Runtime tips:

- Motion-dispersion parameters are only read during startup; use `particle_filter.yaml` (or relaunch with overrides) instead of `ros2 param set` if you need different baselines.
- Debug throttles (`Slip raw…` and `Motion noise scales…`) tell you the active slip ratio and per-axis scale factors—watch them while tuning straight-line launches.
- If the wheel odom stream drops out for longer than `slip_wheel_timeout`, the filter falls back to deriving wheel speed from the EKF delta and logs that the measurement is stale.

# Docs

This code is the staff solution to the lab guide found in the [/docs](/particle_filter/docs) folder. A mathematical derivation of MCL is available in that guide.

There is also documentation on RangeLibc in the [/docs](/particle_filter/docs) folder.

The code itself also contains comments describing purpose of each method.

# Cite

This library accompanies the following [publication](http://arxiv.org/abs/1705.01167).

    @article{walsh17,
        author = {Corey Walsh and
                  Sertac Karaman},
        title  = {CDDT: Fast Approximate 2D Ray Casting for Accelerated Localization},
        volume = {abs/1705.01167},
        url    = {http://arxiv.org/abs/1705.01167},
        year   = {2017}}
