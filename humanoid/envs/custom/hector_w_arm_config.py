from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class HectorFullCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 15
        num_single_obs = 65
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 94 
        # + 187
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 18
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 0.8
        vel_limit = 0.5
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hector_v2/xacro/robot_w_arm.urdf'

        name = "hector"
        foot_name = "toe"
        knee_name = "calf"

        terminate_after_contacts_on = ['base', 'thigh', 'shoulder', 'twist', 'roll']
        penalize_contacts_on = ["base", "thigh"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'L_hip_joint': 0., # hip yaw
            'L_hip_roll_joint': 0.0,  # hip roll
            'L_thigh_joint': 0.785, # hip pitch/thigh
            'L_calf_joint': -1.578,
            'L_toe_joint': 0.785,
            'R_hip_joint': 0.,
            'R_hip_roll_joint': 0.,
            'R_thigh_joint': 0.785,
            'R_calf_joint': -1.578,
            'R_toe_joint': 0.785,
            'L_shoulder_yaw_joint': 0.,
            'L_shoulder_pitch_joint': 0.,
            'L_shoulder_roll_joint': 0.,
            'L_elbow_joint': -0.785,
            'R_shoulder_yaw_joint': 0.,
            'R_shoulder_pitch_joint': 0.,
            'R_shoulder_roll_joint': 0.,
            'R_elbow_joint': -0.785
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_joint': 80.0, 'hip_roll': 80.0, 'thigh': 80.0,'calf': 80.0, 'toe': 60.0, 
                     'shoulder_yaw': 30.0, 'shoulder_pitch': 30.0, 'shoulder_roll': 30.0, 'elbow': 30.0}
        damping = {'hip_joint': 5.0, 'hip_roll': 5.0, 'thigh': 5.0, 'calf': 5.0, 'toe': 3.0,
                   'shoulder_yaw': 3.0, 'shoulder_pitch': 3.0, 'shoulder_roll': 3.0, 'elbow': 3.0}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1., 4.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_delay = 0.0
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.6, 0.8]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.55
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 200  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 0.0
            feet_clearance = 1.2
            feet_contact_number = 1.5
            # gait
            feet_air_time = 1.5 
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.02
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 1.2
            orientation = 1.
            base_height = 0.8
            base_acc = 0.22
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -1e-3
            dof_acc = -1e-6
            collision = -1.

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


class HectorFullCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [768, 512, 128]
        critic_hidden_dims = [768, 768, 768]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 10001  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'hector_arm'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
