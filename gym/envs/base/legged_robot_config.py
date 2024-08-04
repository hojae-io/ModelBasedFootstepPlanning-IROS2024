# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096  # (n_robots in Rudin 2021 paper - batch_size = n_steps * n_robots)
        num_actuators = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        episode_length_s = 20  # episode length in seconds

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x_range = [-0.6, 0.6]
        measured_points_x_num_sample = 13
        measured_points_y_range = [-0.6, 0.6]
        measured_points_y_num_sample = 13 
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        platform_size = 5.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, yaw_vel
        resampling_time = 10. # time before command are changed[s]
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = 1.   # min max [m/s]
            yaw_vel = 1.    # min max [rad/s]

    class init_state:

        # * target state when actuation = 0, also reset positions for basic mode
        default_joint_angles = {"joint_a": 0.,
                                "joint_b": 0.}

        reset_mode = "reset_to_basic" 
        # reset setup chooses how the initial conditions are chosen. 
        # "reset_to_basic" = a single position
        # "reset_to_range" = uniformly random from a range defined below

        # * root defaults
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initial conditiosn for reset_to_range
        dof_pos_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}
        dof_vel_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}
        root_pos_range = [[0., 0.],  # x
                          [0., 0.],  # y
                          [0.5, 0.75],  # z
                          [0., 0.],  # roll
                          [0., 0.],  # pitch
                          [0., 0.]]  # yaw
        root_vel_range = [[-0.1, 0.1],  # x
                          [-0.1, 0.1],  # y
                          [-0.1, 0.1],  # z
                          [-0.1, 0.1],  # roll
                          [-0.1, 0.1],  # pitch
                          [-0.1, 0.1]]  # yaw

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # actuation scale: target angle = actuationScale * actuation + defaultAngle
        actuation_scale = 0.5
        # decimation: Number of control actuation updates @ sim DT per policy DT
        decimation = 4
        exp_avg_decay = None

    class asset:
        file = ""
        keypoints = []
        end_effectors = []
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        disable_actuations = False
        disable_motors = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        rotor_inertia = []

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class weights:
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            lin_vel_z = 0.
            ang_vel_xy = 0.
            orientation = 0.
            torques = 0.
            dof_vel = 0.
            base_height = 0.
            feet_air_time = 0.
            collision = 0.
            feet_stumble = 0.
            actuation_rate = 0.
            actuation_rate2 = 0.
            stand_still = 0.
            dof_pos_limits = 0.
        
        class termination_weights:
            termination = 0.

        curriculum = False
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.  # ! may want to turn this off
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
    
    class scaling:
        base_lin_vel = 2.0
        base_ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05

        commands = 1
        # Action scales
        dof_pos = 1
        dof_pos_target = dof_pos  # scale by range of motion
        clip_actions = 100.

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [1, 0, 4]  # [m]
        lookat = [2., 5, 1.]  # [m]
        record = False

    class sim:
        dt =  0.001
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 10.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotRunnerCfg(BaseConfig):
    seed = 2
    runner_class_name = 'OnPolicyRunner'

    class logging:
        enable_local_saving = True
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        actor_obs = ["observation_a",
                     "observation_b",
                     "these_need_to_be_atributes_(states)_of_the_robot_env"]

        critic_obs = ["observation_x",
                      "observation_y",
                      "critic_obs_can_be_the_same_or_different_than_actor_obs"]

        actions = ["q_des"]
        class noise:
            dof_pos = 0.01
            dof_vel = 0.01
            base_lin_vel = 0.1
            base_ang_vel = 0.2
            projected_gravity = 0.05
            height_measurements = 0.1

        class reward:
            class weights:
                tracking_lin_vel = .0
                tracking_ang_vel = 0.
                lin_vel_z = 0
                ang_vel_xy = 0.
                orientation = 0.
                torques = 0.
                dof_vel = 0.
                base_height = 0.
                collision = 0.
                actuator_rate = 0.
                actuator_rate2 = 0.
                stand_still = 0.
                dof_pos_limits = 0.
            class termination_weights:
                termination = 0.
    class algorithm:
        class PPO:
            # training params
            value_loss_coef = 1.0
            use_clipped_value_loss = True
            clip_param = 0.2
            entropy_coef = 0.01
            num_learning_epochs = 5
            num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
            learning_rate = 1.e-4 # 5.e-4
            schedule = 'adaptive' # could be adaptive, fixed
            gamma = 0.99
            lam = 0.95
            desired_kl = 0.01
            max_grad_norm = 1.


    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration (n_steps in Rudin 2021 paper - batch_size = n_steps * n_robots)
        max_iterations = 1500 # number of policy updates
        SE_learner = None
        # logging
        save_interval = 50 # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'legged_robot'

        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
