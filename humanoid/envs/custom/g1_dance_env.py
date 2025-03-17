from requests.packages import target

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  HumanoidTerrain
from humanoid.envs.custom.motions.motion_loader import MotionLoader




def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class G1DacneFreeEnv(LeggedRobot):
    '''
    G1FreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.demo_time = torch.zeros(self.num_envs, device=self.device)
        self._motion_loader = MotionLoader(motion_file=cfg.env.motion_file, device=self.device)
        self.target_dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.target_body_pos = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.target_body_rot = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.target_body_lin_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.target_body_ang_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.target_keypoint_pos = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.target_keypoint_rot = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.target_keypoint_lin_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.target_keypoint_ang_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.keypoint_pos = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.keypoint_rot = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.keypoint_lin_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.keypoint_ang_vel = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.initial_target_keypoint_pos = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()
        self.get_target_init_pos()


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 7] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos + self.default_dof_pos


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 3] = 0.  # commands
        noise_vec[3: 15] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[15: 27] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[27: 39] = 0.  # previous actions
        noise_vec[39: 42] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[42: 45] = noise_scales.quat * self.obs_scales.quat         # euler x,y
        return noise_vec

    # def sample_motion_data(self):
    #     control_dt = 0.02
    #     # control_dt = self.dt
    #     self.demo_time += control_dt
    #     self.demo_time = torch.where(self.demo_time > self._motion_loader.duration, torch.zeros_like(self.demo_time),
    #                                  self.demo_time)
    #     demo_time_np = self.demo_time.cpu().numpy()
    #     demo_dof_pos, demo_dof_vel, demo_body_pos, demo_body_rot, demo_body_lin_vel, demo_body_ang_vel = \
    #         self._motion_loader.sample(num_samples=self.num_envs, times=demo_time_np)
    #     return demo_dof_pos, demo_dof_vel, demo_body_pos, demo_body_rot, demo_body_lin_vel, demo_body_ang_vel
    def get_target_init_pos(self):
        demo_body_pos = self._motion_loader.sample(num_samples=self.num_envs, times=0.0)[2]
        demo_body_pos = demo_body_pos.unsqueeze(0)
        skip_indices = {7, 17, 25, 26, 34}
        keep_indices = [i for i in range(demo_body_pos.shape[1]) if i not in skip_indices]
        self.initial_target_keypoint_pos = demo_body_pos[:, keep_indices, ...]


    def sample_motion_data(self):
        # control_dt = 0.02
        control_dt = self.dt
        self.demo_time += control_dt
        effective_demo_time = torch.where(self.demo_time < 0.2,
                                          torch.zeros_like(self.demo_time),
                                          self.demo_time - 0.2)
        effective_demo_time = torch.where(effective_demo_time > self._motion_loader.duration,
                                          torch.zeros_like(effective_demo_time),
                                          effective_demo_time)
        demo_time_np = effective_demo_time.cpu().numpy()
        demo_dof_pos, demo_dof_vel, demo_body_pos, demo_body_rot, demo_body_lin_vel, demo_body_ang_vel = \
            self._motion_loader.sample(num_samples=self.num_envs, times=demo_time_np)

        # self.demo_time = torch.where(
        #     self.demo_time > self._motion_loader.duration,
        #     torch.zeros_like(self.demo_time),
        #     self.demo_time
        # )
        # demo_time_np = self.demo_time.cpu().numpy()
        # demo_dof_pos, demo_dof_vel, demo_body_pos, demo_body_rot, demo_body_lin_vel, demo_body_ang_vel = \
        #     self._motion_loader.sample(num_samples=self.num_envs, times=demo_time_np)
        keypoint_names = ['left_shoulder_pitch_link', 'right_shoulder_pitch_link',
                          'left_elbow_link', 'right_elbow_link',
                          'left_wrist_pitch_link', 'right_wrist_pitch_link',
                          'left_ankle_pitch_link', 'right_ankle_pitch_link']
        keypoint_indices = [i for i, name in enumerate(self.body_names) if name in keypoint_names]
        self.keypoint_pos = demo_body_pos[:, keypoint_indices, ...]
        self.keypoint_rot = demo_body_rot[:, keypoint_indices, ...]
        self.keypoint_lin_vel = demo_body_lin_vel[:, keypoint_indices, ...]
        self.keypoint_ang_vel = demo_body_ang_vel[:, keypoint_indices, ...]

        target_keypoint_indices = [5, 12, 18, 21, 23, 27, 30, 32]
        self.target_keypoint_indices = target_keypoint_indices
        self.target_keypoint_pos = demo_body_pos[:, target_keypoint_indices, ...]
        self.target_keypoint_rot = demo_body_rot[:, target_keypoint_indices, ...]
        self.target_keypoint_lin_vel = demo_body_lin_vel[:, target_keypoint_indices, ...]
        self.target_keypoint_ang_vel = demo_body_ang_vel[:, target_keypoint_indices, ...]

        # remove_names = ['pelvis_contour_link', 'head_link', 'left_rubber_hand', 'logo_link', 'right_rubber_hand']
        # keep_indices = [i for i, name in enumerate(self.body_names) if name not in remove_names]
        skip_indices = {7, 17, 25, 26, 34}
        keep_indices = [i for i in range(demo_body_pos.shape[1]) if i not in skip_indices]

        demo_body_pos = demo_body_pos[:, keep_indices, ...]
        demo_body_rot = demo_body_rot[:, keep_indices, ...]
        demo_body_lin_vel = demo_body_lin_vel[:, keep_indices, ...]
        demo_body_ang_vel = demo_body_ang_vel[:, keep_indices, ...]

        demo_body_pos = demo_body_pos + self.init_rigid_states[:, :, 0:3] - self.initial_target_keypoint_pos
        # demo_body_pos = self.initial_rigid_states[:, :, 0:3]

        return demo_dof_pos, demo_dof_vel, demo_body_pos, demo_body_rot, demo_body_lin_vel, demo_body_ang_vel

    def step(self, actions):
        # demo_dof_pos, demo_dof_vel, demo_body_pos, demo_body_rot, demo_body_lin_vel, demo_body_ang_vel = self.sample_motion_data()
        # # self.demo_body_pos = demo_body_pos
        # self.target_dof_pos = demo_dof_pos[:, -29:]
        # self.target_body_pos = demo_body_pos
        # self.target_body_rot = demo_body_rot
        # self.target_body_lin_vel = demo_body_lin_vel
        # self.target_body_ang_vel = demo_body_ang_vel
        # target_root_quat = self.target_body_rot[:, 0, :]
        # self.target_root_euler = get_euler_xyz_tensor(target_root_quat)

        if self.cfg.env.use_ref_actions:
            actions += self.ref_action

        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # print(actions)
        # dynamic randomization
        # print(self.dof_pos)
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        return super().step(actions)


    def compute_observations(self):

        demo_dof_pos, demo_dof_vel, demo_body_pos, demo_body_rot, demo_body_lin_vel, demo_body_ang_vel = self.sample_motion_data()
        # self.demo_body_pos = demo_body_pos
        self.target_dof_pos = demo_dof_pos[:, -29:]
        self.target_dof_vel = demo_dof_vel[:, -29:]
        # self.default_dof_pos = self.target_dof_pos #make action base on target_dof_pos
        self.target_body_pos = demo_body_pos
        self.target_body_rot = demo_body_rot
        self.target_body_lin_vel = demo_body_lin_vel
        self.target_body_ang_vel = demo_body_ang_vel
        target_root_quat = self.target_body_rot[:, 0, :]
        self.target_root_euler = get_euler_xyz_tensor(target_root_quat)
        phase = self._get_phase()
        self.torso_state = self.rigid_state[:, self.torso_idx]
        self.torso_projected_gravity = quat_rotate_inverse(self.torso_state[:, 3:7], self.gravity_vec)
        torso_quat = self.torso_state[:, 3:7]
        torso_euler = get_euler_xyz_tensor(torso_quat)

        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 1.

        self.command_input = self.commands[:, :3] * self.commands_scale
        
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        # (self.dof_pos - self.last_dof_pos) * self.obs_scales.dof_vel / self.dt

        # if self.cfg.domain_rand.randomize_obs_motor_latency:
        #     self.obs_motor = self.obs_motor_latency_buffer[torch.arange(self.num_envs), :, self.obs_motor_latency_simstep.long()]
        # else:
        #     self.obs_motor = torch.cat((q, dq), 1)

        # if self.cfg.domain_rand.randomize_obs_imu_latency:
        #     self.obs_imu = self.obs_imu_latency_buffer[torch.arange(self.num_envs), :, self.obs_imu_latency_simstep.long()]
        # else:              
        #     self.obs_imu = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, self.base_euler_xyz * self.obs_scales.quat), 1)
        
        diff = self.dof_pos - self.default_dof_pos
        # print(self.dof_pos - self.last_dof_pos)
        self.privileged_obs_buf = torch.cat((
            # self.command_input,
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # 12
            self.dof_pos, # 12
            dq,  # 12
            self.actions,  # 12
            # diff,  # 10
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            torch.flatten(self.rigid_state[:, self.feet_indices, :3], start_dim=1, end_dim=2),
            torch.flatten(self.rigid_state[:, self.feet_indices, 7:10], start_dim=1, end_dim=2),
            self.root_states[:, :3], # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            # self.cfg.domain_rand.action_noise, # 1
            self.body_mass,  # 1
            stance_mask,  # 2
            contact_mask,  # 2
            self.target_dof_pos, #29
            self.target_keypoint_pos.view(-1, 24), #24
            (self.keypoint_pos-self.target_keypoint_pos).view(-1, 24), #24
            self.target_dof_vel,  # 29
        ), dim=-1)

        obs_buf = torch.cat((
            # self.command_input, #3
            q,    # 12D
            dq,  # 12D
            self.actions,   # 12D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.target_dof_pos,  # 29
            # self.target_keypoint_pos.view(-1, 24),  # 24
            # (self.keypoint_pos - self.target_keypoint_pos).view(-1, 24),  # 24
            # self.target_dof_vel,  # 29
        ), dim=-1)
        # print(self.measured_heights.size())
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.privileged_obs_buf = torch.cat((self.command_input,
        #         (self.dof_pos - self.default_joint_pd_target) * \
        #         self.obs_scales.dof_pos,  # 12
        #         dq * self.obs_scales.dof_vel,  # 12
        #         self.actions,  # 12
        #         self.base_lin_vel * self.obs_scales.lin_vel,  # 3
        #         self.base_ang_vel * self.obs_scales.ang_vel,  # 3
        #         self.base_euler_xyz * self.obs_scales.quat,  # 3
        #         torch.flatten(self.rigid_state[:, self.feet_indices, :3], start_dim=1, end_dim=2),
        #         torch.flatten(self.rigid_state[:, self.feet_indices, 7:10], start_dim=1, end_dim=2),
        #         self.root_states[:, :3], # 3
        #         self.rand_push_force[:, :2],  # 2
        #         self.rand_push_torque,  # 3
        #         self.env_frictions,  # 1
        #         self.body_mass / 30.,  # 1
        #         # stance_mask,  # 2
        #         contact_mask,
        #         self._get_heights()), dim=-1)

        # print(self.privileged_obs_buf.size())
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)


        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.demo_time[env_ids] = 0.  # reset demo states
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.p_gains_multiplier
        d_gains = self.d_gains * self.d_gains_multiplier
        # torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains * self.dof_vel
        torques = p_gains * (actions_scaled + self.target_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains * self.dof_vel#chage to use target_dof_pos as default
        # (self.dof_pos - self.last_dof_pos)/self.dt

        # print(self.dof_pos)
        # print(self.dof_vel)
        # self.dof_vel

        torques *= self.torque_multiplier

        # torques *= self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        torso_offset = torch.norm(self.target_body_pos[:, self.torso_idx, :] - self.torso_state[:, :3], dim=1)
        self.reset_buf |= torso_offset > 1.5
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

# ================================================ Rewards ================================================== #
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_slippage(self):
        foot_vel = self._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = torch.norm(self.rigid_state[:, self.feet_indices, 7:10]) > 0.1
        # print(torch.norm(self.rigid_state[:, self.feet_indices, 7:9]))
        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos
    
    # def _reward_default_joint_pos(self):
    #     """
    #     Calculates the reward for keeping joint positions close to default positions, with a focus
    #     on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
    #     """
    #     joint_diff = self.dof_pos - self.default_joint_pd_target
    #     joint_diff[:, 0] = 0
    #     joint_diff[:, 6] = 0
    #     joint_diff[:, 4] = 0
    #     joint_diff[:, 10] = 0
    #     left_yaw_roll = joint_diff[:, 1:3]
    #     right_yaw_roll = joint_diff[:,7:9]
    #     yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    #     yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    #     return torch.exp(-yaw_roll * 2.0) - 0.01 * torch.norm(joint_diff, dim=1)
    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        # joint_diff = self.dof_pos - self.default_joint_pd_target
        joint_diff = self.dof_pos - self.target_dof_pos
        joint_diff[:, 0] = 0
        joint_diff[:, 6] = 0
        joint_diff[:, 4] = 0
        joint_diff[:, 10] = 0
        left_yaw_roll = joint_diff[:, 1:3]
        right_yaw_roll = joint_diff[:,7:9]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 2.0) - 0.01 * torch.norm(joint_diff, dim=1)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square((self.dof_pos - self.last_dof_pos) / self.dt), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_lower_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, :12] - self.actions[:, :12]), dim=1)

    def _reward_upper_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, 12:] - self.actions[:, 12:]), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        # return self.reset_buf * ~self.time_out_buf
        remaining_time = self._motion_loader.duration - self.demo_time
        termination_penalty = self.reset_buf * ~self.time_out_buf * (1 + 5 / (remaining_time + 1))
        return termination_penalty
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits).clip(min=0.), dim=1)

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # print(self.commands[:, :2])
        # print(self.base_lin_vel[:, :2])
        # print(lin_vel_error)
        # print(torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma))
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # print(contact)
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        
        rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 #no reward for zero command9
        # rew_airTime *= torch.logical_not(torch.logical_and(torch.norm(self.base_lin_vel[:, :2], dim=1) > 0.05, torch.abs(self.base_ang_vel[:, 2]) > 0.05)) #no reward when robot is not moving
        self.feet_air_time *= ~contact_filt
        # print(rew_airTime)
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_dof_position(self):
        """
        DoF Position Reward:
        r_dof = 3.0 * exp( -0.7 * sum(|q_ref - q|) )
        dof_pos, target_dof_pos: shape (N, D)
        """
        error = torch.sum(torch.abs(self.target_dof_pos - self.dof_pos), dim=1)  # (N,)
        return torch.exp(-0.7 * error)

    def _reward_torso_position(self):
        error = torch.norm(self.target_body_pos[:, self.torso_idx, :] - self.torso_state[:, :3],
                                    dim=1)  # (N, 3)
        error = torch.mean(error)
        reward = torch.exp(-0.7 * error)
        return reward

    def _reward_torso_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.torso_projected_gravity[:, :2]), dim=1)

    def _reward_dof_vel_tracking(self):
        diff_dof_vel = self.target_dof_vel - self.dof_vel
        # scale the diff by self.cfg.rewards.teleop_joint_pos_selection
        # for joint_name, scale in self.cfg.rewards.teleop_joint_pos_selection.items():
        #     joint_index = self.dof_names.index(joint_name)
        #     assert joint_index >= 0, f"Joint {joint_name} not found in the robot"
        #     diff_dof_vel[:, joint_index] *= scale ** .5
        diff_dof_vel_dist = torch.mean(torch.square(diff_dof_vel), dim=1)
        return torch.exp(-0.7* diff_dof_vel_dist)


    def _reward_keypoint_position(self):
        """
        Keypoint Position Reward:
        r_kp = 2.0 * exp( -||p_ref - p|| )
        keypoint_pos, target_keypoint_pos: shape (N, num_bodies, 3)
        loss pelvis_contour_link, head_link, left_rubber_hand, logo_link, right_rubber_hand
        """
        # keypoint_indices = -17
        # keypoint_indices = 0
        # body_rot = self.rigid_state[:, keypoint_indices:, 9:13]
        # self.keypoint_pos - self.target_keypoint_pos
        # error = torch.norm(self.keypoint_pos - self.target_keypoint_pos, dim=2)  # (N, num_bodies, 3)
        error = torch.norm(self.target_body_pos[:, :, :] - self.rigid_state[:, :, :3], dim=2)  # (N, num_bodies, 3)
        lower_error_mean = torch.mean(error[:, :17], dim=1)
        upper_error_mean = torch.mean(error[:, 17:], dim=1)
        reward_lower = torch.exp(-0.5 * lower_error_mean)
        reward_upper = torch.exp(-1.2 * upper_error_mean)
        reward = reward_lower + reward_upper
        return reward
        # error = torch.mean(error, dim=1)  # shape: (N,)
        # return torch.exp(-0.7*error)

    def _reward_lin_velocity(self):
        """
        Linear Velocity Reward:
        r_linVel = 6.0 * exp( -4.0 * ||v_ref - v|| )
        base_lin_vel, target_lin_vel: shape (N, 3)
        """
        target_base_lin_vel = self.target_body_lin_vel[:, 0, :].squeeze(1)
        error = torch.norm(target_base_lin_vel - self.base_lin_vel, dim=1)  # (N,)
        return torch.exp(-4.0 * error)

    def _reward_ang_velocity(self):
        """F
        Linear Velocity Reward:
        r_linVel = 6.0 * exp( -4.0 * ||v_ref - v|| )
        base_lin_vel, target_lin_vel: shape (N, 3)
        """
        target_base_ang_vel = self.target_body_ang_vel[:, 0, :].squeeze(1)
        error = torch.norm(target_base_ang_vel - self.base_ang_vel, dim=1)  # (N,)
        return torch.exp(-4.0 * error)

    def _reward_vel_direction(self):
        """
        Velocity Direction Reward:
        r_dir = 6.0 * exp( -4.0 * (1 - cos(v_ref, v)) )
        """
        eps = 1e-8
        target_base_lin_vel = self.target_body_lin_vel[:, 0, :].squeeze(1)
        dot = torch.sum(target_base_lin_vel * self.base_lin_vel, dim=1)  # (N,)
        norm_target = torch.norm(target_base_lin_vel, dim=1)  # (N,)
        norm_base = torch.norm(self.base_lin_vel, dim=1)  # (N,)
        cos_angle = dot / (norm_target * norm_base + eps)
        return torch.exp(-4.0 * (1 - cos_angle))

    def _reward_roll_pitch(self):
        """
        Roll & Pitch Reward:
        r_rp = 1.0 * exp( - (|roll_ref - roll| + |pitch_ref - pitch|) )
        roll_pitch, target_roll_pitch: shape (N, 2)
        """
        error = torch.sum(torch.abs(self.target_root_euler[:, :2] - self.base_euler_xyz[:, :2]), dim=1)  # (N,)
        return torch.exp(-error)

    def _reward_yaw(self):
        """
        Yaw Reward:
        r_yaw = 1.0 * exp( -|yaw_ref - yaw| )
        yaw, target_yaw: shape (N,)
        """
        error = torch.abs(self.target_root_euler[:, 2] - self.base_euler_xyz[:, 2])  # (N,)
        return torch.exp(-error)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)


