from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2wCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 15
        num_single_obs = 57
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 98
        # + 187
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 16
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 0.8
        vel_limit = 0.7
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w/urdf/go2w.urdf'

        name = "go2w"
        foot_name = "foot"
        knee_name = "calf"

        terminate_after_contacts_on = ['base', 'thigh', 'hip', 'calf', 'Head_lower']
        # terminate_after_contacts_on = ['base']
        # penalize_contacts_on = ["base", "thigh"]
        penalize_contacts_on = ["thigh", "calf"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = True
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 4.
        terrain_width = 4.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 1.0    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.34]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,
            'RL_hip_joint': 0.,
            'FR_hip_joint': -0.,
            'RR_hip_joint': -0.,

            'FL_thigh_joint': 1.,
            'RL_thigh_joint': 1.,
            'FR_thigh_joint': 1.,
            'RR_thigh_joint': 1.,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,

            'FL_foot_joint': 0,
            'RL_foot_joint': 0,
            'FR_foot_joint': 0,
            'RR_foot_joint': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_joint': 50, "thigh_joint": 50, "calf_joint": 50, "foot_joint": 0}
        damping = {'hip_joint': 2, "thigh_joint": 2, "calf_joint": 2, "foot_joint": 0.5}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20  # 50hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.02  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**22  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.2]
        randomize_base_mass = True
        added_mass_range = [-2., 4.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6
        # dynamic randomization
        action_delay = 0.0
        action_noise = 0.08
        motor_strength = [0.8, 1.1]

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 2.5]   # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.38
        min_dist = 0.1
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 0.25
        max_contact_force = 180  # Forces above this value are penalized

        class scales:
            termination = -200.
            default_joint_pos = 2.0
            foot_slip = -0.
            feet_clearance = 0.0
            tracking_lin_vel = 4.5
            tracking_ang_vel = 3.8
            ang_vel_xy = -0.8
            torques = -8.e-10
            dof_acc = -5.e-9
            lin_vel_z = -2.5
            feet_air_time = 1.
            orientation = -1.0
            dof_pos_limits = -1.
            base_height = -5.0
            no_fly = 0.0
            dof_vel = -2.e-10
            feet_contact_forces = -0.01

            action_rate = -0.01

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100
        clip_actions = 100


class Go2wCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [768, 768, 768]
        critic_hidden_dims = [768, 768, 768]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
        schedule = 'adaptive'
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 10001  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'go2w'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
