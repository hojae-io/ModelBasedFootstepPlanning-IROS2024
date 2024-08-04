"""
Environment file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gym.envs.humanoid.humanoid_vanilla_config import HumanoidVanillaCfg
from gym.utils.math import *
from gym.envs import LeggedRobot
from isaacgym import gymapi, gymutil
from .humanoid_utils import VelCommandGeometry
from gym.utils import VanillaKeyboardInterface
from .jacobian import apply_coupling


class HumanoidVanilla(LeggedRobot):
    cfg: HumanoidVanillaCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _setup_keyboard_interface(self):
        self.keyboard_interface = VanillaKeyboardInterface(self)

    def _init_buffers(self):
        super()._init_buffers()
        self.base_height = self.root_states[:, 2:3]
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.eps = 0.2
        self.phase_freq = 1.

        # * Observation variables
        self.phase_sin = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.phase_cos = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_schedule = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_contact = torch.zeros(self.num_envs, len(self.feet_ids), dtype=torch.bool, device=self.device, requires_grad=False) # contacts on right & left sole
        self.foot_states = torch.zeros(self.num_envs, len(self.feet_ids), 7, dtype=torch.float, device=self.device, requires_grad=False) # num_envs x (left & right foot) x (x, y, z, quat)    
        self.foot_heading = torch.zeros(self.num_envs, len(self.feet_ids), dtype=torch.float, device=self.device, requires_grad=False) # num_envs x (left & right foot heading)  
        self.base_heading = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.step_period = self.cfg.commands.step_period
        self.full_step_period = 2*self.cfg.commands.step_period
        self.phase_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False) # Number of transition since the beginning of the episode
        self.update_phase_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False) # envs whose step commands are updated
        
        self.current_step = torch.zeros(self.num_envs, len(self.feet_ids), 3, dtype=torch.float, device=self.device, requires_grad=False) # (left & right foot) x (x, y, heading) wrt base x,y-coordinate
        self.step_length = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) # step length
        self.step_width = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) # step width
        self.dstep_length = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) # desired step length
        self.dstep_width = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) # desired step width
        self.CoM = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.total_weight = self.mass_total * (-self.cfg.sim.gravity[2])

        if self.cfg.terrain.measure_heights:
            self.measured_heights_obs = torch.zeros_like(self.measured_heights)  

    def _compute_torques(self):
        self.desired_pos_target = self.dof_pos_target + self.default_dof_pos
        q = self.dof_pos.clone()
        qd = self.dof_vel.clone()
        q_des = self.desired_pos_target.clone()
        qd_des = torch.zeros_like(self.dof_pos_target)
        tau_ff = torch.zeros_like(self.dof_pos_target)
        kp = self.p_gains.clone()
        kd = self.d_gains.clone()

        if (self.cfg.asset.apply_humanoid_jacobian):
            torques = apply_coupling(q, qd, q_des, qd_des, kp,
                                      kd, tau_ff)
        else:
            torques = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff

        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

        return torques.view(self.torques.shape)       
      
    def _reset_system(self, env_ids):
        super()._reset_system(env_ids)
        if self.cfg.commands.resampling_time == -1:
            self.commands[env_ids, :] = 0.
        self.phase_count[env_ids] = 0 
        self.update_phase_ids[env_ids] = False
        self.phase[env_ids] = 0 
        self.foot_states[env_ids] = self.rigid_body_state[:, self.feet_ids, :7][env_ids]
        self.current_step[env_ids] = torch.tensor([[0., 0.15, 0.], [0., -0.15, 0.]], device=self.device, requires_grad=False) # current_step initializatoin

    def _post_physics_step_callback(self):
        self.phase_count += 1
        self.phase += 1 / self.full_step_period
        
        self.update_phase_ids = (self.phase_count >= self.full_step_period) 
        self.phase[self.update_phase_ids] = 0
        self.phase_count[self.update_phase_ids] = 0

        self._update_robot_states()
        self._calculate_CoM()

        env_ids = (
            self.episode_length_buf
            % int(self.cfg.commands.resampling_time / self.dt) == 0) \
            .nonzero(as_tuple=False).flatten()
        if self.cfg.commands.resampling_time == -1 :
            # print(self.commands)
            pass  # when the joystick is used, the self.commands variables are overridden
        else:
            self._resample_commands(env_ids)
            if (self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0)):
                self._push_robots()
                
    def _update_robot_states(self):
        """ Update robot state variables """
        self.foot_states = self.rigid_body_state[:, self.feet_ids, :7]  
        forward = quat_apply(self.base_quat, self.forward_vec)
        self.base_heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1) 
        self.foot_contact = torch.gt(self.contact_forces[:, self.feet_ids, 2], 0)

        right_foot_forward = quat_apply(self.foot_states[:,0,3:7], self.forward_vec)
        left_foot_forward = quat_apply(self.foot_states[:,1,3:7], self.forward_vec)
        right_foot_heading = wrap_to_pi(torch.atan2(right_foot_forward[:, 1], right_foot_forward[:, 0])) 
        left_foot_heading = wrap_to_pi(torch.atan2(left_foot_forward[:, 1], left_foot_forward[:, 0]))
        self.foot_heading[:,0] = right_foot_heading
        self.foot_heading[:,1] = left_foot_heading

        # * Update current step
        current_step_masked = self.current_step[self.foot_contact]
        current_step_masked[:, :2] = self.foot_states[self.foot_contact][:,:2]
        current_step_masked[:, 2] = self.foot_heading[self.foot_contact]
        self.current_step[self.foot_contact] = current_step_masked

        # * Calculate step length and width
        theta = torch.atan2(self.commands[:,1:2], self.commands[:,0:1])

        rright_foot_pos_x = torch.cos(theta)*self.current_step[:,0,0:1] + torch.sin(theta)*self.current_step[:,0,1:2]
        rright_foot_pos_y = -torch.sin(theta)*self.current_step[:,0,0:1] + torch.cos(theta)*self.current_step[:,0,1:2]
        rleft_foot_pos_x = torch.cos(theta)*self.current_step[:,1,0:1] + torch.sin(theta)*self.current_step[:,1,1:2]
        rleft_foot_pos_y = -torch.sin(theta)*self.current_step[:,1,0:1] + torch.cos(theta)*self.current_step[:,1,1:2]

        self.step_length = torch.abs(rright_foot_pos_x - rleft_foot_pos_x)
        self.step_width = torch.abs(rright_foot_pos_y - rleft_foot_pos_y)

    def _calculate_CoM(self):
        """ Calculates the Center of Mass of the robot """
        self.CoM = (self.rigid_body_state[:,:,:3] * self.rigid_body_mass.unsqueeze(1)).sum(dim=1) / self.mass_total
    
    def _set_obs_variables(self):
        self.base_height[:] = self.root_states[:, 2:3]
        self.contact_schedule = self.smooth_sqr_wave(self.phase)
        self.phase_sin = torch.sin(2*torch.pi*self.phase)
        self.phase_cos = torch.cos(2*torch.pi*self.phase)

        self.base_lin_vel_world = self.root_states[:, 7:10].clone()

        if self.cfg.terrain.measure_heights:
            self.measured_heights_obs = torch.clip(self.measured_heights, -1, 1.)
            
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # Termination for contact
        term_contact = torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1)
        self.terminated = torch.any((term_contact > 1.), dim=1)

        # Termination for velocities, orientation, and low height
        self.terminated |= torch.any(
          torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 10., dim=1)
        self.terminated |= torch.any(
          torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 5., dim=1)
        self.terminated |= torch.any(
          torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)
        self.terminated |= torch.any(
          torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)
        self.terminated |= torch.any(self.base_pos[:, 2:3] < 0.3, dim=1)

        # Termination for wrong steps (not on stepping stones)
        if self.cfg.terrain.measure_heights:
            self.terminated |= ((self.foot_states[:,:,2] < 0).sum(dim=1) != 0)

        # # no terminal reward for time-outs
        self.timed_out = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.terminated | self.timed_out

    def _visualization(self):
        self.gym.clear_lines(self.viewer)
        self._draw_heightmap_vis()
        # self._draw_velocity_arrow_vis()
        self._draw_world_velocity_arrow_vis()

    def _draw_velocity_arrow_vis(self):
        """ Draws linear / angular velocity arrow for humanoid 
            Angular velocity is described by axis-angle representation """
        origins = self.base_pos + quat_apply(self.base_quat, torch.tensor([0.,0.,.5]).repeat(self.num_envs, 1).to(self.device))
        lin_vel_command = quat_apply(self.base_quat, torch.cat((self.commands[:, :2], torch.zeros((self.num_envs,1), device=self.device)), dim=1)/5)
        ang_vel_command = quat_apply(self.base_quat, torch.cat((torch.zeros((self.num_envs,2), device=self.device), self.commands[:, 2:3]), dim=1)/5)
        for i in range(self.num_envs):
            lin_vel_arrow = VelCommandGeometry(origins[i], lin_vel_command[i], color=(0,1,0))
            ang_vel_arrow = VelCommandGeometry(origins[i], ang_vel_command[i], color=(0,1,0))
            gymutil.draw_lines(lin_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None)
            gymutil.draw_lines(ang_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None)   

    def _draw_world_velocity_arrow_vis(self):
        """ Draws linear / angular velocity arrow for humanoid 
            Angular velocity is described by axis-angle representation """
        origins = self.base_pos + quat_apply(self.base_quat, torch.tensor([0.,0.,.5]).repeat(self.num_envs, 1).to(self.device))
        lin_vel_command = torch.cat((self.commands[:, :2], torch.zeros((self.num_envs,1), device=self.device)), dim=1)/5
        # ang_vel_command = quat_apply(self.base_quat, torch.cat((torch.zeros((self.num_envs,2), device=self.device), self.commands[:, 2:3]), dim=1)/5)
        for i in range(self.num_envs):
            lin_vel_arrow = VelCommandGeometry(origins[i], lin_vel_command[i], color=(0,1,0))
            # ang_vel_arrow = VelCommandGeometry(origins[i], ang_vel_command[i], color=(0,1,0))
            gymutil.draw_lines(lin_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None)
            # gymutil.draw_lines(ang_vel_arrow, self.gym, self.viewer, self.envs[i], pose=None)

# ########################## REWARDS ######################## #

    # * "True" rewards * #

    def _reward_tracking_lin_vel(self):
        # Reward tracking specified linear velocity command
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        return self._negsqrd_exp(error, a=1.).sum(dim=1)
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Reward tracking yaw angular velocity command
        ang_vel_error = torch.square(
            (self.commands[:, 2] - self.base_ang_vel[:, 2])*2/torch.pi)
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_lin_vel_world(self):
        # Reward tracking linear velocity command in world frame
        error = self.commands[:, :2] - self.root_states[:, 7:9]
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        return self._negsqrd_exp(error, a=1.).sum(dim=1)
    
    # * Shaping rewards * #

    def _reward_base_height(self):
        # Reward tracking specified base height
        error = (self.cfg.rewards.base_height_target - self.base_height).flatten()
        return self._negsqrd_exp(error)

    def _reward_base_heading(self):
        # Reward tracking desired base heading
        command_heading = torch.atan2(self.commands[:, 1], self.commands[:, 0])
        base_heading_error = torch.abs(wrap_to_pi(command_heading - self.base_heading.squeeze(1)))

        return self._neg_exp(base_heading_error, a=torch.pi/2)
        return self._negsqrd_exp(base_heading_error, a=torch.pi/4)
    
    def _reward_base_z_orientation(self):
        # Reward tracking upright orientation
        error = torch.norm(self.projected_gravity[:, :2], dim=1)
        return self._negsqrd_exp(error, a=0.2)

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.

        # Yaw joints regularization around 0
        error += self._negsqrd_exp(
            (self.dof_pos[:, 0]) / self.scales['dof_pos'])
        error += self._negsqrd_exp(
            (self.dof_pos[:, 5]) / self.scales['dof_pos'])

        # Ab/ad joint symmetry
        # error += self._negsqrd_exp(
        #     (self.dof_pos[:, 1] - self.dof_pos[:, 6])
        #     / self.scales['dof_pos'])
        error += self._negsqrd_exp((self.dof_pos[:, 1]) / self.scales['dof_pos'])
        error += self._negsqrd_exp((self.dof_pos[:, 6]) / self.scales['dof_pos'])

        # Pitch joint symmetry
        # error += self._negsqrd_exp(
        #     (self.dof_pos[:, 2] + self.dof_pos[:, 7])
        #     / self.scales['dof_pos'])
        # error += self._negsqrd_exp((self.dof_pos[:, 3] - self.dof_pos[:, 8]) / self.scales['dof_pos']) # knee

        return error/4

    def _reward_contact_schedule(self):
        """ Alternate right and left foot contacts
            First, right foot contacts (left foot swing), then left foot contacts (right foot swing) """ 
        return (self.foot_contact[:,0].int() - self.foot_contact[:,1].int()) * self.contact_schedule.squeeze(1)


# ##################### HELPER FUNCTIONS ################################## #

    def smooth_sqr_wave(self, phase):
        p = 2.*torch.pi*phase * self.phase_freq
        return torch.sin(p) / torch.sqrt(torch.sin(p)**2. + self.eps**2.)
            

    def original_smooth_sqr_wave(self, phase):
        p = 2.*torch.pi*phase * self.phase_freq
        return torch.sin(p) / \
            (2*torch.sqrt(torch.sin(p)**2. + self.eps**2.)) + 1./2.
    