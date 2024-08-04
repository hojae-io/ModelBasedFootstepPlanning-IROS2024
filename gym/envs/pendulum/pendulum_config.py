from gym import LEGGED_GYM_ENVS_DIR
from gym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotRunnerCfg
import torch

class PendulumCfg(FixedRobotCfg):

    class env(FixedRobotCfg.env):
        num_envs = 4096 # 1096
        num_observations = 2
        num_actions = 1
        episode_length_s = 5
        num_critic_obs = num_observations

    class init_state(FixedRobotCfg.init_state):

        default_joint_angles = {"actuator": 0.}
        
        # * default setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"
        # reset_mode = "reset_to_basic"

        # * initial conditions for reset_to_range
        dof_pos_range = {'actuator': [-torch.pi, torch.pi]}
        dof_vel_range = {'actuator': [-1., 1.]}
        
    class control(FixedRobotCfg.control):
        # * PD Drive parameters:
        stiffness = {'actuator': 10.}  # [N*m/rad]
        damping = {'actuator': 0.5}  # [N*m*s/rad]

        actuated_joints_mask = [1,  # slider_to_cart
                                0]  # cart_to_pole

        decimation = 2
        # ctrl_frequency = 250
        # desired_sim_frequency = 500

    class asset(FixedRobotCfg.asset):
        # Things that differ
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pendulum/urdf/pendulum.urdf"
        flip_visual_attachments = False

        # Toggles to keep
        disable_gravity = False
        disable_motors = False  # all torques set to 0

    class rewards(FixedRobotCfg.rewards):
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.  # ! may want to turn this off

        class weights:
            pendulum_vel = 0.05
            torques = 0.1
            upright_pendulum = 30
        class termination_weights:
            termination = 0.

    class normalization(FixedRobotCfg.normalization):
        class obs_scales:
            dof_pos = 1. #[1/3., 1/torch.pi]
            dof_vel = 1. #[1/20., 1/(4*torch.pi)]

    class noise(FixedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.001
            dof_vel = 0.01
    
    class sim(FixedRobotCfg.sim):
        dt =  0.002

class PendulumRunnerCfg(FixedRobotRunnerCfg):
    # We need random experiments to run
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy(FixedRobotRunnerCfg.policy):
        init_noise_std = 1.0
        log_std_bounds = [-20., 2.]
        num_layers = 2
        num_units = 64
        actor_hidden_dims = [num_units] * num_layers
        critic_hidden_dims = [num_units] * num_layers
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        actions_limits = [-1.5, 1.5]

    class algorithm(FixedRobotRunnerCfg.algorithm):
        class PPO:
            # training params
            value_loss_coef = 1.0
            use_clipped_value_loss = True
            clip_param = 0.2
            entropy_coef = 0.01
            num_learning_epochs = 5
            num_mini_batches = 4 # * mini batch size = num_envs*nsteps / nminibatches
            learning_rate = 1.e-3
            schedule = 'adaptive'  # could be adaptive, fixed
            gamma = 0.998
            lam = 0.95
            desired_kl = 0.01
            max_grad_norm = 1.
        class SAC:
            # algorithm training hyperparameters
            actor_learning_rate = 1e-3
            critic_learning_rate = 1e-3
            entropy_learning_rate = 1e-3
            initial_entropy_value = 0.2
            gamma = 0.99                    # discount factor
            polyak = 0.005                  # soft update hyperparameter
            num_learning_epochs = 5
            num_mini_batches = 4
            mini_batch_size = 1024 #1024
            target_entropy = None
            max_grad_norm = 1.
            target_update_period = 1
            replay_buffer_size_per_env = 500 # ratio of new data = num_steps_per_env / replay_buffer_size_per_env
            custom_initialization = True # xavier initialization
            initial_random_exploration = True

    class runner(FixedRobotRunnerCfg.runner):
        policy_class_name = 'ActorCritic'
        # algorithm_class_name = 'PPO'
        algorithm_class_name = 'SAC'
        num_steps_per_env = 32  # per iteration
        max_iterations = 500  # number of policy updates

        # * logging
        # * check for potential saves every this many iterations
        save_interval = 100
        run_name = ''
        experiment_name = 'pendulum'

        # * load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
