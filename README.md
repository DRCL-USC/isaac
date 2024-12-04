
## Usage Guide

#### Examples

```bash
# Launching PPO Policy Training for 'v1' Across 4096 Environments
# This command initiates the PPO algorithm-based training for the humanoid task.
python scripts/train.py --task=humanoid_ppo --run_name v1 --headless --num_envs 4096

# Evaluating the Trained PPO Policy 'v1'
# This command loads the 'v1' policy for performance assessment in its environment. 
# Additionally, it automatically exports a JIT model, suitable for deployment purposes.
python scripts/play.py --task=humanoid_ppo --run_name v1

# Implementing Simulation-to-Simulation Model Transformation
# This command facilitates a sim-to-sim transformation using exported 'v1' policy.
python scripts/sim2sim.py --load_model /path/to/logs/XBot_ppo/exported/policies/policy_1.pt

# Run our trained policy
python scripts/sim2sim.py --load_model /path/to/logs/XBot_ppo/exported/policies/policy_example.pt

```

#### 1. Default Tasks


- **humanoid_ppo**
   - Purpose: Baseline, PPO policy, Multi-frame low-level control
   - Observation Space: Variable $(47 \times H)$ dimensions, where $H$ is the number of frames
   - $[O_{t-H} ... O_t]$
   - Privileged Information: $73$ dimensions

- **humanoid_dwl (coming soon)**

#### 2. PPO Policy
- **Training Command**: For training the PPO policy, execute:
  ```
  python humanoid/scripts/train.py --task=humanoid_ppo --load_run log_file_path --name run_name
  ```
- **Running a Trained Policy**: To deploy a trained PPO policy, use:
  ```
  python humanoid/scripts/play.py --task=humanoid_ppo --load_run log_file_path --name run_name
  ```
- By default, the latest model of the last run from the experiment folder is loaded. However, other run iterations/models can be selected by adjusting `load_run` and `checkpoint` in the training config.


#### 4. Parameters
- **Task**: To run for different robots, set `--task` to desired one. Available options are: humanoid_ppo, hector, go2w
- **CPU and GPU Usage**: To run simulations on the CPU, set both `--sim_device=cpu` and `--rl_device=cpu`. For GPU operations, specify `--sim_device=cuda:{0,1,2...}` and `--rl_device={0,1,2...}` accordingly. Please note that `CUDA_VISIBLE_DEVICES` is not applicable, and it's essential to match the `--sim_device` and `--rl_device` settings.
- **Headless Operation**: Include `--headless` for operations without rendering.
- **Rendering Control**: Press 'v' to toggle rendering during training.
- **Policy Location**: Trained policies are saved in `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`.

#### 5. Command-Line Arguments
For RL training, please refer to `humanoid/utils/helpers.py#L161`.
For the sim-to-sim process, please refer to `humanoid/scripts/sim2sim.py#L169`.



## Citation

Please cite the following if you use this code or parts of it:
```
@article{gu2024humanoid,
  title={Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer},
  author={Gu, Xinyang and Wang, Yen-Jen and Chen, Jianyu},
  journal={arXiv preprint arXiv:2404.05695},
  year={2024}
}

@inproceedings{gu2024advancing,
  title={Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning},
  author={Gu, Xinyang and Wang, Yen-Jen and Zhu, Xiang and Shi, Chengming and Guo, Yanjiang and Liu, Yichen and Chen, Jianyu},
  booktitle={Robotics: Science and Systems},
  year={2024},
  url={https://enriquecoronadozu.github.io/rssproceedings2024/rss20/p058.pdf}
}
```

## Acknowledgment

The implementation of Humanoid-Gym relies on resources from [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl) projects, created by the Robotic Systems Lab. We specifically utilize the `LeggedRobot` implementation from their research to enhance our codebase.

## Any Questions?

If you have any more questions, please contact [support@robotera.com](mailto:support@robotera.com) or create an issue in this repository.
