# SPDX-License-Identifier: BSD-3-Clause
import os
import cv2
import copy
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 8
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs_save_file = os.path.join(LEGGED_GYM_ROOT_DIR, 'observations', 'all_observations.txt')
    os.makedirs(os.path.dirname(obs_save_file), exist_ok=True)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to:', path)

        torch.onnx.export(
            ppo_runner.alg.actor_critic.actor,
            obs,
            "locomotion_net.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['obs'],
            output_names=['action'],
            dynamic_axes={'input': {0: '615'}, 'output': {0: '10'}}
        )

    logger = Logger(env.dt)
    robot_index = -1 # which robot is used for logging
    joint_index = 9 # which joint is used for logging
    stop_state_log = 1200 # number of steps before plotting states

    if RENDER:
        # 创建相机和视频写入器的列表
        cameras = []
        videos = []

        # 为每个机器人创建一个相机和视频写入器
        for i in range(env_cfg.env.num_envs):
            camera_properties = gymapi.CameraProperties()
            camera_properties.width = 1920
            camera_properties.height = 1080
            h1 = env.gym.create_camera_sensor(env.envs[i], camera_properties)
            camera_offset = gymapi.Vec3(1, -1, 0.5)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
            actor_handle = env.gym.get_actor_handle(env.envs[i], 0)
            body_handle = env.gym.get_actor_rigid_body_handle(env.envs[i], actor_handle, 0)
            env.gym.attach_camera_to_body(
                h1, env.envs[i], body_handle,
                gymapi.Transform(camera_offset, camera_rotation),
                gymapi.FOLLOW_POSITION
            )
            cameras.append(h1)

            # 创建视频文件
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
            experiment_dir = os.path.join(video_dir, train_cfg.runner.experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            video_path = os.path.join(experiment_dir, f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{args.load_run}_agent_{i}.mp4")
            video = cv2.VideoWriter(video_path, fourcc, 50.0, (1920, 1080))
            videos.append(video)

    for i in tqdm(range(stop_state_log)):
        actions = policy(obs.detach())

        if FIX_COMMAND:
            env.commands[:, 0] = 0.5
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            env.gym.sync_frame_time(env.sim)

            # 录制每个相机的图像
            for j, h1 in enumerate(cameras):
                img = env.gym.get_camera_image(env.sim, env.envs[j], h1, gymapi.IMAGE_COLOR)
                img = np.reshape(img, (1080, 1920, 4))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                videos[j].write(img[..., :3])

        logger.log_states(
            {
                'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                'dof_torque': env.torques[robot_index, joint_index].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            }
        )

        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)

    logger.print_rewards()
    logger.plot_states()

    if RENDER:
        for video in videos:
            video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    play(args)
