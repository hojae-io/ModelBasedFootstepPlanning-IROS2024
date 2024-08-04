import torch

class Augmentor:
    def __init__(self, augmentation_toggles):
        # self.cfg = cfg
        # self.augmentations = self._get_augmentations_from_cfg(cfg)
        # self.do_kin = 'kinematics' in augmentation_toggles  # add_kinematics_augmentations
        # self.do_jac = 'jacobian' in augmentation_toggles  # cfg.env.obs_augmentations.add_jacobian_augmentations
        # self.do_cen = 'centripetal' in augmentation_toggles  # cfg.env.obs_augmentations.add_centripetal_augmentations
        # self.do_cor = 'coriolis' in augmentation_toggles  # cfg.env.obs_augmentations.add_coriolis_augmentations

        self.add_kinematics = 'kinematics' in augmentation_toggles
        self.add_mass_matrix = 'coriolis' in augmentation_toggles
        self.add_coriolis_matrix = 'coriolis' in augmentation_toggles

        self.dof_pos_names  = ['rf_haa', 'rf_hfe', 'rf_hke',
                              'lf_haa', 'lf_hfe', 'lf_hke',
                              'rh_haa', 'rh_hfe', 'rh_hke',
                              'lh_haa', 'lh_hfe', 'lh_hke']
        self.body_lin_vel_names = ['d_x', 'd_y', 'd_z']
        self.body_ang_vel_names = ['d_roll', 'd_pitch', 'd_yaw']
        self.dof_vel_names  = ['d_rf_haa', 'd_rf_hfe', 'd_rf_hke',
                              'd_lf_haa', 'd_lf_hfe', 'd_lf_hke',
                              'd_rh_haa', 'd_rh_hfe', 'd_rh_hke',
                              'd_lh_haa', 'd_lh_hfe', 'd_lh_hke']

        self.legs = ['rf_', 'lf_', 'rh_', 'lh_']

        self.num_augmentations = 0

        self.first_obs_buf_idx = 0

        self.all_dof_names = self.dof_pos_names + self.body_lin_vel_names + self.body_ang_vel_names + self.dof_vel_names

    def set_first_idx_in_obs_buf(self, idx):
        self.first_obs_buf_idx = idx

    def _get_augmentations_from_cfg(self, cfg):
        augmentations = []
        if cfg.env.obs_augmentations.add_kinematics_augmentations:
            augmentations += cfg.env.obs_augmentations.kinematic_augmentations
        if cfg.env.obs_augmentations.add_jacobian_augmentations:
            augmentations += cfg.env.obs_augmentations.jacobian_augmentations
        if cfg.env.obs_augmentations.add_centripetal_augmentations:
            augmentations += cfg.env.obs_augmentations.centripetal_augmentations
        if cfg.env.obs_augmentations.add_coriolis_augmentations:
            augmentations += cfg.env.obs_augmentations.coriolis_augmentations
        return augmentations

    def get_number_augmentations(self):
        return len(self.augmentations)

    def write_jit_script(self, file_name):
        pass

    def apply_augmentations(self, obs_buf, body_lin_vel, body_ang_vel, dof_pos, dof_vel):
        if self.add_kinematics:
            pass
        if self.add_mass_matrix:
            pass
        if self.add_coriolis_matrix:
            pass


    # def apply_augmentations(self, body_lin_vel, body_ang_vel, dof_pos, dof_vel):
    #     applied_augmentations_list = []
    #     if self.do_kin:
    #         # Official number of augmentations: 6*4 = 24
    #         for leg in ['rf_', 'lf_', 'rh_', 'lh_']:
    #             idx_of_abad = self.dof_pos_names.index(f'{leg}haa')
    #             idx_of_hip = self.dof_pos_names.index(f'{leg}hfe')
    #             idx_of_knee = self.dof_pos_names.index(f'{leg}hke')
    #
    #             abad = dof_pos[:, idx_of_abad]
    #             hip = dof_pos[:, idx_of_hip]
    #             knee = dof_pos[:, idx_of_knee]
    #
    #             # sin(hip + knee)
    #             applied_augmentations_list.append(torch.sin(hip + knee).unsqueeze(dim=-1))
    #             # sin(hip)
    #             applied_augmentations_list.append(torch.sin(hip).unsqueeze(dim=-1))
    #             # sin(abad) * cos(hip + knee)
    #             applied_augmentations_list.append((torch.sin(abad) * torch.cos(hip + knee)).unsqueeze(dim=-1))
    #             # sin(abad) * cos(hip)
    #             applied_augmentations_list.append((torch.sin(abad) * torch.cos(hip)).unsqueeze(dim=-1))
    #             # cos(abad) * cos(hip + knee)
    #             applied_augmentations_list.append((torch.cos(abad) * torch.cos(hip + knee)).unsqueeze(dim=-1))
    #             # cos(abad) * cos(hip)
    #             applied_augmentations_list.append((torch.cos(abad) * torch.cos(hip)).unsqueeze(dim=-1))
    #
    #     if self.do_jac:
    #         # Official number of augmentations: 6*4 = 24
    #         for leg in ['rf_', 'lf_', 'rh_', 'lh_']:
    #             idx_of_abad = self.dof_pos_names.index(f'{leg}haa')
    #             idx_of_hip = self.dof_pos_names.index(f'{leg}hfe')
    #             idx_of_knee = self.dof_pos_names.index(f'{leg}hke')
    #
    #             abad = dof_pos[:, idx_of_abad]
    #             hip = dof_pos[:, idx_of_hip]
    #             knee = dof_pos[:, idx_of_knee]
    #
    #             # cos(hip + knee)
    #             applied_augmentations_list.append(torch.cos(hip + knee).unsqueeze(dim=-1))
    #             # cos(hip)
    #             applied_augmentations_list.append(torch.cos(hip).unsqueeze(dim=-1))
    #             # cos(hip + knee)
    #             # DONE
    #             # cos(abad)*cos(hip + knee)
    #             # DONE
    #             # cos(abad)*cos(hip)
    #             # DONE
    #             # sin(abad)*sin(hip + knee)
    #             applied_augmentations_list.append((torch.sin(abad) * torch.sin(hip + knee)).unsqueeze(dim=-1))
    #             # sin(abad)*sin(hip)
    #             applied_augmentations_list.append((torch.sin(abad) * torch.sin(hip)).unsqueeze(dim=-1))
    #             # sin(abad)*sin(hip + knee)
    #             # DONE
    #             # sin(abad)*cos(hip + knee)
    #             # DONE
    #             # sin(abad)*cos(hip)
    #             # DONE
    #             # cos(abad)*sin(hip + knee)
    #             applied_augmentations_list.append((torch.cos(abad) * torch.sin(hip + knee)).unsqueeze(dim=-1))
    #             # cos(abad)*sin(hip)
    #             applied_augmentations_list.append((torch.cos(abad) * torch.sin(hip)).unsqueeze(dim=-1))
    #             # cos(abad)*sin(hip + knee)
    #             # DONE
    #
    #     if self.do_cen:
    #         # Official number of augmentations: 6 + 3*4 = 18
    #         d_x = body_lin_vel[:, 0]
    #         d_y = body_lin_vel[:, 1]
    #         d_z = body_lin_vel[:, 2]
    #         d_roll = body_ang_vel[:, 0]
    #         d_pitch = body_ang_vel[:, 1]
    #         d_yaw = body_ang_vel[:, 2]
    #
    #         sqr_scaling = 1e-4 / 2
    #
    #         applied_augmentations_list.append((d_x * d_x * sqr_scaling).unsqueeze(dim=-1))
    #         applied_augmentations_list.append((d_y * d_y * sqr_scaling).unsqueeze(dim=-1))
    #         applied_augmentations_list.append((d_z * d_z * sqr_scaling).unsqueeze(dim=-1))
    #         applied_augmentations_list.append((d_roll * d_roll * sqr_scaling).unsqueeze(dim=-1))
    #         applied_augmentations_list.append((d_pitch * d_pitch * sqr_scaling).unsqueeze(dim=-1))
    #         applied_augmentations_list.append((d_yaw * d_yaw * sqr_scaling).unsqueeze(dim=-1))
    #
    #         for leg in self.legs:
    #             idx_of_abad = self.dof_vel_names.index(f'd_{leg}haa')
    #             idx_of_hip = self.dof_vel_names.index(f'd_{leg}hfe')
    #             idx_of_knee = self.dof_vel_names.index(f'd_{leg}hke')
    #
    #             d_abad = dof_vel[:, idx_of_abad]
    #             d_hip = dof_vel[:, idx_of_hip]
    #             d_knee = dof_vel[:, idx_of_knee]
    #
    #             applied_augmentations_list.append((d_abad * d_abad * sqr_scaling).unsqueeze(dim=-1))
    #             applied_augmentations_list.append((d_hip * d_hip * sqr_scaling).unsqueeze(dim=-1))
    #             applied_augmentations_list.append((d_knee * d_knee * sqr_scaling).unsqueeze(dim=-1))
    #
    #     if self.do_cor:
    #         # Official number of augmentations: 6 + 3*4 = 18
    #         d_x = body_lin_vel[:, 0]
    #         d_y = body_lin_vel[:, 1]
    #         d_z = body_lin_vel[:, 2]
    #         d_roll = body_ang_vel[:, 0]
    #         d_pitch = body_ang_vel[:, 1]
    #         d_yaw = body_ang_vel[:, 2]
    #         all_velocities = [d_x, d_y, d_z, d_roll, d_pitch, d_yaw]
    #
    #         for leg in self.legs:
    #             idx_of_abad = self.dof_vel_names.index(f'd_{leg}haa')
    #             idx_of_hip = self.dof_vel_names.index(f'd_{leg}hfe')
    #             idx_of_knee = self.dof_vel_names.index(f'd_{leg}hke')
    #             all_velocities.append(dof_vel[:, idx_of_abad])
    #             all_velocities.append(dof_vel[:, idx_of_hip])
    #             all_velocities.append(dof_vel[:, idx_of_knee])
    #
    #         cross_scaling = 1e-3 / 2
    #
    #         for idx in range(len(all_velocities)):
    #             for jdx in range(idx, len(all_velocities)):
    #                 if idx != jdx:
    #                     applied_augmentations_list.append((all_velocities[idx] * all_velocities[jdx] * cross_scaling).unsqueeze(dim=-1))
    #
    #     return applied_augmentations_list
    #
