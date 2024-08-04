"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from gym.envs.base.legged_robot_config \
    import LeggedRobotCfg, LeggedRobotRunnerCfg


class HumanoidVanillaCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actuators = 10
        episode_length_s = 5 # 100

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = 'plane' # 'plane' 'heightfield' 'trimesh'
        measure_heights = False # True, False
        selected = True # True, False
        # terrain_kwargs = {'type': 'stepping_stones'}
        # terrain_kwargs = {'type': 'random_uniform'}
        terrain_kwargs = {'type': 'gap'}
        # difficulty = 0.35 # For gap terrain
        # platform_size = 5.5 # For gap terrain
        difficulty = 5.0 # For rough terrain
        terrain_length = 18. # For rough terrain
        terrain_width = 18. # For rough terrain
        # terrain types: [pyramid_sloped, random_uniform, stairs down, stairs up, discrete obstacles, stepping_stones, gap, pit]
        terrain_proportions = [0., 0.5, 0., 0., 0., 0.5, 0.]

    # class init_state(LeggedRobotCfg.init_state):
    #     reset_mode = 'reset_to_range'
    #     pos = [0., 0., 0.62]        # x,y,z [m]
    #     rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
    #     lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
    #     ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

    #     # ranges for [x, y, z, roll, pitch, yaw]
    #     root_pos_range = [
    #         [0., 0.],  # x
    #         [0., 0.],  # y
    #         [0.62, 0.62],  # z
    #         [-torch.pi/10, torch.pi/10],  # roll
    #         [-torch.pi/10, torch.pi/10],  # pitch
    #         [-torch.pi/10, torch.pi/10]   # yaw
    #     ]

    #     # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
    #     root_vel_range = [
    #         [-.5, .5],  # x
    #         [-.5, .5],  # y
    #         [-.5, .5],  # z
    #         [-.5, .5],  # roll
    #         [-.5, .5],  # pitch
    #         [-.5, .5]   # yaw
    #     ]

    #     default_joint_angles = {
    #         '01_right_hip_yaw': 0.,
    #         '02_right_hip_abad': 0.1,
    #         '03_right_hip_pitch': -0.667751,
    #         '04_right_knee': 1.4087,  # 0.6
    #         '05_right_ankle': -0.708876,
    #         '06_left_hip_yaw': 0.,
    #         '07_left_hip_abad': 0.1,
    #         '08_left_hip_pitch': -0.667751,
    #         '09_left_knee': 1.4087,  # 0.6
    #         '10_left_ankle': -0.708876,
    #     }

    #     dof_pos_range = {
    #         '01_right_hip_yaw': [-0.1, 0.1],
    #         '02_right_hip_abad': [-0.1, 0.3],
    #         '03_right_hip_pitch': [-0.8, -0.4],
    #         '04_right_knee': [1.3, 1.5],
    #         '05_right_ankle': [-0.9, -0.5],
    #         '06_left_hip_yaw': [-0.1, 0.1],
    #         '07_left_hip_abad': [-0.1, 0.3],
    #         '08_left_hip_pitch': [-0.8, -0.4],
    #         '09_left_knee': [1.3, 1.5],
    #         '10_left_ankle': [-0.9, -0.5],
    #     }

    #     dof_vel_range = {
    #         '01_right_hip_yaw': [-0.1, 0.1],
    #         '02_right_hip_abad': [-0.1, 0.1],
    #         '03_right_hip_pitch': [-0.1, 0.1],
    #         '04_right_knee': [-0.1, 0.1],
    #         '05_right_ankle': [-0.1, 0.1],
    #         '06_left_hip_yaw': [-0.1, 0.1],
    #         '07_left_hip_abad': [-0.1, 0.1],
    #         '08_left_hip_pitch': [-0.1, 0.1],
    #         '09_left_knee': [-0.1, 0.1],
    #         '10_left_ankle': [-0.1, 0.1],
    #     }

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = 'reset_to_range'
        pos = [0., 0., 0.77]        # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0., 0.],  # x
            [0., 0.],  # y
            [0.77, 0.77],  # z
            [-torch.pi/20, torch.pi/20],  # roll
            [-torch.pi/20, torch.pi/20],  # pitch
            [-torch.pi/20, torch.pi/20]   # yaw
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-.1, .1],  # x
            [-.1, .1],  # y
            [-.1, .1],  # z
            [-.1, .1],  # roll
            [-.1, .1],  # pitch
            [-.1, .1]   # yaw
        ]

        default_joint_angles = {
            '01_right_hip_yaw': 0.,
            '02_right_hip_abad': 0.,
            '03_right_hip_pitch': -0.2,
            '04_right_knee': 0.6,
            '05_right_ankle': 0.,
            '06_left_hip_yaw': 0.,
            '07_left_hip_abad': 0.,
            '08_left_hip_pitch': -0.2,
            '09_left_knee': 0.6,
            '10_left_ankle': 0.,
        }

        dof_pos_range = {
            '01_right_hip_yaw': [-0.1, 0.1],
            '02_right_hip_abad': [-0.1, 0.1],
            '03_right_hip_pitch': [-0.2, 0.2],
            '04_right_knee': [0.6, 0.7],
            '05_right_ankle': [-0.3, 0.0],
            '06_left_hip_yaw': [-0.1, 0.1],
            '07_left_hip_abad': [-0.1, 0.1],
            '08_left_hip_pitch': [-0.2, 0.2],
            '09_left_knee': [0.6, 0.7],
            '10_left_ankle': [-0.3, 0.0],
        }

        dof_vel_range = {
            '01_right_hip_yaw': [-0.1, 0.1],
            '02_right_hip_abad': [-0.1, 0.1],
            '03_right_hip_pitch': [-0.1, 0.1],
            '04_right_knee': [-0.1, 0.1],
            '05_right_ankle': [-0.1, 0.1],
            '06_left_hip_yaw': [-0.1, 0.1],
            '07_left_hip_abad': [-0.1, 0.1],
            '08_left_hip_pitch': [-0.1, 0.1],
            '09_left_knee': [-0.1, 0.1],
            '10_left_ankle': [-0.1, 0.1],
        }


    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints

        stiffness = {
            '01_right_hip_yaw': 30.,
            '02_right_hip_abad': 30.,
            '03_right_hip_pitch': 30.,
            '04_right_knee': 30.,
            '05_right_ankle': 30.,
            '06_left_hip_yaw': 30.,
            '07_left_hip_abad': 30.,
            '08_left_hip_pitch': 30.,
            '09_left_knee': 30.,
            '10_left_ankle': 30.,
        }
        damping = {
            '01_right_hip_yaw': 3.,
            '02_right_hip_abad': 3.,
            '03_right_hip_pitch': 3.,
            '04_right_knee': 3.,
            '05_right_ankle': 3.,
            '06_left_hip_yaw': 3.,
            '07_left_hip_abad': 3.,
            '08_left_hip_pitch': 3.,
            '09_left_knee': 3.,
            '10_left_ankle': 3.
        }

        actuation_scale = 1.0
        exp_avg_decay = None
        decimation = 10

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 3        # write them out here
        resampling_time = 10.
        step_period = 35 # 35

        class ranges:
            # TRAINING COMMAND RANGES #
            # lin_vel_x = [-2., 2.]        # min max [m/s]
            # lin_vel_y = .75   # min max [m/s]
            # yaw_vel = 1.     # min max [rad/s]

            lin_vel_x = [-2.0, 2.0] # [-2.0, 2.0] # [-3.0, 3.0] # min max [m/s]
            lin_vel_y = 2. # 2. # 1.5   # min max [m/s]
            yaw_vel = 0.    # min max [rad/s]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]

        randomize_base_mass = False
        added_mass_range = [-1., 1.]

        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0.5

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/humanoid/urdf/humanoid_fixed_arms_sf_update.urdf'
        keypoints = ["base"]
        end_effectors = ['right_foot', 'left_foot']
        foot_name = 'foot'
        terminate_after_contacts_on = [
            'base',
            'right_upper_leg',
            'right_lower_leg',
            'left_upper_leg',
            'left_lower_leg',
            'right_upper_arm',
            'right_lower_arm',
            'right_hand',
            'left_upper_arm',
            'left_lower_arm',
            'left_hand',
        ]

        disable_gravity = False
        disable_actions = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = False
        flip_visual_attachments = False

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

        angular_damping = 0.1
        rotor_inertia = [
            0.01188,    # RIGHT LEG
            0.01188,
            0.01980,
            0.07920,
            0.04752,
            0.01188,    # LEFT LEG
            0.01188,
            0.01980,
            0.07920,
            0.04752,
        ]

        apply_humanoid_jacobian = False # True, False

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.62
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 1500.

        # negative total rewards clipped at zero (avoids early termination)
        only_positive_rewards = False
        tracking_sigma = 0.25
        
        class weights(LeggedRobotCfg.rewards.weights):
            # * Regularization rewards * #
            actuation_rate = 1e-3
            actuation_rate2 = 1e-4
            torques = 1e-4
            dof_vel = 1e-3
            lin_vel_z = 1e-1
            ang_vel_xy = 1e-2
            dof_pos_limits = 10
            torque_limits = 1e-2

            # * Floating base rewards * #
            base_height = 1.
            base_heading = 3.
            base_z_orientation = 1.
            tracking_lin_vel_world = 4.

            # tracking_lin_vel = 5.
            # tracking_ang_vel = 2.

            # * Stepping rewards * #
            joint_regularization = 1.
            contact_schedule = 3.

        class termination_weights(LeggedRobotCfg.rewards.termination_weights):
            termination = 1.

    class scaling(LeggedRobotCfg.scaling):
        base_height = 1. # 1./0.6565
        base_lin_vel = 1.
        base_ang_vel = 1.
        projected_gravity = 1.
        dof_pos = 1.
        dof_vel = 1.

        commands = 1
        # Action scales
        dof_pos = 1
        dof_pos_target = dof_pos  # scale by range of motion
        
        clip_actions = 10.


class HumanoidVanillaRunnerCfg(LeggedRobotRunnerCfg):
    do_wandb = True
    seed = -1
    class policy(LeggedRobotRunnerCfg.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = 'elu'
        normalize_obs = True # True, False

        actor_obs = ["base_height",
                     "base_lin_vel_world", # "base_lin_vel",
                     "base_heading",
                     "base_ang_vel",
                     "projected_gravity",
                     "commands",
                     "phase_sin",
                     "phase_cos",
                     "dof_pos",
                     "dof_vel",]

        critic_obs = actor_obs

        actions = ["dof_pos_target"]
        class noise:
            base_height = 0.05
            base_lin_vel = 0.05
            base_lin_vel_world = 0.05
            base_heading = 0.01
            base_ang_vel = 0.05
            projected_gravity = 0.05
            dof_pos = 0.05
            dof_vel = 0.5
            measured_heights_obs = 0.1

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        class PPO:
            # algorithm training hyperparameters
            value_loss_coef = 1.0
            use_clipped_value_loss = True
            clip_param = 0.2
            entropy_coef = 0.01
            num_learning_epochs = 5
            num_mini_batches = 4    # minibatch size = num_envs*nsteps/nminibatches
            learning_rate = 1.e-5
            schedule = 'adaptive'   # could be adaptive, fixed
            gamma = 0.99
            lam = 0.95
            desired_kl = 0.01
            max_grad_norm = 1.

    class runner(LeggedRobotRunnerCfg.runner):
        num_steps_per_env = 24
        max_iterations = 1000
        run_name = 'full'
        experiment_name = 'Humanoid_Vanilla'
        save_interval = 100
        plot_input_gradients = False
        plot_parameter_gradients = False


