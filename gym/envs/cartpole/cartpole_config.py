import torch
from gym.envs.base.fixed_robot_config import FixedRobotCfg, FixedRobotRunnerCfg

class CartpoleCfg(FixedRobotCfg):
    class env(FixedRobotCfg.env):
        num_envs = 4096 # 1096
        num_observations = 4
        num_actions = 1  # 1 for the cart force
        episode_length_s = 5 # 100
        num_critic_obs = num_observations

    class init_state(FixedRobotCfg.init_state):

        default_joint_angles = {"slider_to_cart": 0.,
                                "cart_to_pole": 0.}

        # default setup chooses how the initial conditions are chosen. 
        # "reset_to_basic" = a single position
        # "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range" 

        # * initial conditiosn for reset_to_range
        dof_pos_range = {'slider_to_cart': [-2.5, 2.5],
                         'cart_to_pole': [-torch.pi, torch.pi]}
        dof_vel_range = {'slider_to_cart': [-1., 1.],
                         'cart_to_pole': [-1., 1.]}

    class control(FixedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'slider_to_cart': 10.}  # [N*m/rad]
        damping = {'slider_to_cart': 0.5}  # [N*m*s/rad]

        # for each dof: 1 if actuated, 0 if passive
        # Empty implies no chance in the _compute_torques step
        actuated_joints_mask = [1,  # slider_to_cart
                                0]  # cart_to_pole

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2
        # ctrl_frequency = 250
        # desired_sim_frequency = 500

    class asset(FixedRobotCfg.asset):
        # Things that differ
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/cartpole/urdf/cartpole.urdf"
        flip_visual_attachments = False

        # Toggles to keep
        disable_gravity = False
        disable_motors = False  # all torques set to 0

    class rewards(FixedRobotCfg.rewards):
        class weights:
            cart_vel = 0.05
            pole_vel = 0.05
            torques = 0.1
            upright_pole = 30
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

class CartpoleRunnerCfg(FixedRobotRunnerCfg):
    # We need random experiments to run
    seed = -1
    runner_class_name = 'OnPolicyRunner'

    class policy(FixedRobotRunnerCfg.policy):
        init_noise_std = 1.0
        num_layers = 2
        num_units = 64
        actor_hidden_dims = [num_units] * num_layers
        critic_hidden_dims = [num_units] * num_layers
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(FixedRobotRunnerCfg.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.998
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(FixedRobotRunnerCfg.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 96  # per iteration
        max_iterations = 500  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'cartpole'

        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
