import os
from gym import LEGGED_GYM_ROOT_DIR
from isaacgym import gymapi
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from PIL import Image

def moving_average(x, w):
    """ Moving average filter 
        x: input signal
        w: window size
    """
    nan = np.array([np.nan] * (w-1))
    mvag = np.convolve(x, np.ones(w), 'valid') / w
    return np.concatenate((nan, mvag))

class ScreenShotter():

    def __init__(self, env, experiment_name, log_dir):
        self.env = env
        self.experiment_name = experiment_name
        self.run_name = os.path.basename(log_dir)
        self.folderpath = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.experiment_name, 'analysis')
        os.makedirs(self.folderpath, exist_ok=True)
        self.screenshot_cnt = 0

    def screenshot(self, image):
        # Create a PIL Image from our NumPy array
        img = Image.fromarray(image, 'RGB')
        # Save the image as PDF
        filepath = os.path.join(self.folderpath, f"{self.run_name}_{self.screenshot_cnt}.pdf")
        img.save(filepath, "PDF", resolution=100.0)
        self.screenshot_cnt += 1

class AnalysisRecorder():

    def __init__(self, env, experiment_name, log_dir):
        self.env = env
        self.experiment_name = experiment_name
        self.run_name = os.path.basename(log_dir)
        self.frames = []
        self.states_dict = defaultdict(list) # defaultdict(lambda: defaultdict(list)), defaultdict(list)
        self.commands_dict = defaultdict(list)
        self.fps = 100.
        self.episode_length = 0
        self.folderpath = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.experiment_name, 'analysis')
        os.makedirs(self.folderpath, exist_ok=True)
        self.screenshot_cnt = 0

    def log(self, image, states_dict, commands_dict):
        # * Log images
        self.frames.append(image)

        # * Log states and commands
        for key, value in states_dict.items():
            if key == 'root_states':
                self.states_dict['COM_vel_x'].append(value[7].item())
                self.states_dict['COM_vel_y'].append(value[8].item())
            elif key == 'step_length':
                self.states_dict['step_length'].append(value.item())
            elif key == 'step_width':
                self.states_dict['step_width'].append(value.item())
            elif key == "contact_forces":
                self.states_dict['rf_contact_forces_x'].append(value[self.env.feet_ids[0],0].item())
                self.states_dict['rf_contact_forces_y'].append(value[self.env.feet_ids[0],1].item())
                self.states_dict['rf_contact_forces_z'].append(value[self.env.feet_ids[0],2].item())
                self.states_dict['lf_contact_forces_x'].append(value[self.env.feet_ids[1],0].item())
                self.states_dict['lf_contact_forces_y'].append(value[self.env.feet_ids[1],1].item())
                self.states_dict['lf_contact_forces_z'].append(value[self.env.feet_ids[1],2].item())

        for key, value in commands_dict.items():
            if key == 'commands':
                self.commands_dict['COM_dvel_x'].append(value[0].item())
                self.commands_dict['COM_dvel_y'].append(value[1].item())
            elif key == 'dstep_length':
                self.commands_dict['dstep_length'].append(value.item())
            elif key == 'dstep_width':
                self.commands_dict['dstep_width'].append(value.item())
            elif key == 'step_commands':
                self.commands_dict['dstep_left_x'].append(value[0,0].item())
                self.commands_dict['dstep_left_y'].append(value[0,1].item())
                self.commands_dict['dstep_right_x'].append(value[1,0].item())
                self.commands_dict['dstep_right_y'].append(value[1,1].item())

        self.episode_length += 1

    def save_and_exit(self):
        # self.make_animation()
        self.make_contact_forces_animation()
        self.make_video()
        exit()

    def make_animation(self):
        """ Make animation for states_dict and commands_dict using FuncAnimation from matplotlib """
        print("Creating animation...")
        episode_len = len(self.states_dict['COM_vel_x'])
        fig = plt.figure(figsize=(10, 10))
        spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2.5, 1])

        # * Define the animation for COM tracking
        ax = fig.add_subplot(spec[0])

        def COM_vel_2D_init():
            COM_vel_x_ani.set_data(np.linspace(0, 1, 0), COM_vel_x[0:0])
            COM_vel_y_ani.set_data(np.linspace(0, 1, 0), COM_vel_y[0:0])
            return [COM_vel_x_ani, COM_vel_y_ani]

        def COM_vel_2D_update(i):
            COM_vel_x_ani.set_data(np.linspace(0, i-1, i), COM_vel_x[0:i])
            COM_vel_y_ani.set_data(np.linspace(0, i-1, i), COM_vel_y[0:i])
            COM_dvel_x_ani.set_data(np.linspace(0, i-1, i), COM_dvel_x[0:i])
            COM_dvel_y_ani.set_data(np.linspace(0, i-1, i), COM_dvel_y[0:i])
            return [COM_vel_x_ani, COM_vel_y_ani, ax]

        COM_vel_x = self.states_dict['COM_vel_x']
        COM_vel_y = self.states_dict['COM_vel_y']
        COM_dvel_x = self.commands_dict['COM_dvel_x']
        COM_dvel_y = self.commands_dict['COM_dvel_y']
        ax.set_xlim(0, episode_len)
        ax.set_ylim(min(min(COM_vel_x), min(COM_vel_y), min(COM_dvel_x), min(COM_dvel_y))-0.1, max(max(COM_vel_x), max(COM_vel_y), max(COM_dvel_x), max(COM_dvel_y))+0.1)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('CoM velocity (m/s)')
        tick_loc = np.arange(0, episode_len, 100)
        tick_labels = tick_loc / 100
        ax.xaxis.set_ticks(tick_loc)
        ax.xaxis.set_ticklabels(tick_labels)
        ax.grid(ls='--')

        COM_vel_x_ani, = ax.plot([], [], color='k', label='CoM velocity x')
        COM_dvel_x_ani, = ax.plot([], [], color='k', linestyle='--', label='desired CoM velocity x')
        COM_vel_y_ani, = ax.plot([], [], color='purple', label='CoM velocity y')
        COM_dvel_y_ani, = ax.plot([], [], color='purple', linestyle='--', label='desired CoM velocity y')
        ax.legend(loc='upper right')

        # COM_vel_2D = FuncAnimation(fig=fig, init_func=COM_vel_2D_init, func=COM_vel_2D_update, frames=range(episode_len), interval=50, blit=False)

        # * Define the animation for step length and step width
        bx = fig.add_subplot(spec[1])

        step_length = self.states_dict['step_length']
        step_width = self.states_dict['step_width']
        dstep_length = self.commands_dict['dstep_length']
        dstep_width = self.commands_dict['dstep_width']

        def step_params_2D_init():
            step_length_ani.set_data(np.linspace(0, 1, 0), step_length[0:0])
            step_width_ani.set_data(np.linspace(0, 1, 0), step_width[0:0])
            dstep_length_ani.set_data(np.linspace(0, 1, 0), dstep_length[0:0])
            dstep_width_ani.set_data(np.linspace(0, 1, 0), dstep_width[0:0])
            return [step_length_ani, step_width_ani]

        def step_params_2D_update(i):
            step_length_ani.set_data(np.linspace(0, i-1, i), step_length[0:i])
            step_width_ani.set_data(np.linspace(0, i-1, i), step_width[0:i])
            dstep_length_ani.set_data(np.linspace(0, i-1, i), dstep_length[0:i])
            dstep_width_ani.set_data(np.linspace(0, i-1, i), dstep_width[0:i])
            return [step_length_ani, step_width_ani, bx]

        bx.set_xlim(0, episode_len)
        bx.set_ylim(min(min(step_length), min(step_width), min(dstep_length), min(dstep_width))-0.1, max(max(step_length), max(step_width), max(dstep_length), max(dstep_width))+0.1)
        bx.set_xlabel('time (s)')
        bx.set_ylabel('step length/width (m)')
        tick_loc = np.arange(0, episode_len, 100)
        tick_labels = tick_loc / 100
        bx.xaxis.set_ticks(tick_loc)
        bx.xaxis.set_ticklabels(tick_labels)
        bx.grid(ls='--')

        step_length_ani, = bx.plot([], [], color='gray', label='step length')
        dstep_length_ani, = bx.plot([], [], color='gray', linestyle='--', label='desired step length')
        step_width_ani, = bx.plot([], [], color='cyan', label='step width')
        dstep_width_ani, = bx.plot([], [], color='cyan', linestyle='--', label='desired step width')
        bx.legend(loc='upper right')

        # step_params_2D = FuncAnimation(fig=fig, init_func=step_params_2D_init, func=step_params_2D_update, frames=range(episode_len), interval=50, blit=False)

        # * Combine all the animations
        def _init_func():
            artist1 = COM_vel_2D_init()
            artist2 = step_params_2D_init()
            return artist1 + artist2
        
        def _update_func(i):
            artist1 = COM_vel_2D_update(i)
            artist2 = step_params_2D_update(i)
            return artist1 + artist2

        anim = FuncAnimation(fig=fig, init_func=_init_func, func=_update_func, frames=range(episode_len), interval=50, blit=False)

        # * Save the animation
        filepath = os.path.join(self.folderpath, f"{self.run_name}_plot.mp4")
        # COM_vel_2D.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])
        # step_params_2D.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])
        anim.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])

        print(f"Animation saved to {filepath}")

    def make_contact_forces_animation(self):
        """ Make animation for contact forces using FuncAnimation from matplotlib """
        print("Creating contact forces animation...")
        fig = plt.figure(figsize=(10, 10))
        spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])

        # * Define the animation for right foot contact forces
        ax = fig.add_subplot(spec[0])

        def rf_contact_forces_2D_init():
            rf_contact_forces_x_ani.set_data(np.linspace(0, 1, 0), rf_contact_forces_x[0:0])
            rf_contact_forces_y_ani.set_data(np.linspace(0, 1, 0), rf_contact_forces_y[0:0])
            rf_contact_forces_z_ani.set_data(np.linspace(0, 1, 0), rf_contact_forces_z[0:0])
            return [rf_contact_forces_x_ani, rf_contact_forces_y_ani, rf_contact_forces_z_ani, total_weight_line] 

        def rf_contact_forces_2D_update(i):
            rf_contact_forces_x_ani.set_data(np.linspace(0, i-1, i), rf_contact_forces_x[0:i])
            rf_contact_forces_y_ani.set_data(np.linspace(0, i-1, i), rf_contact_forces_y[0:i])
            rf_contact_forces_z_ani.set_data(np.linspace(0, i-1, i), rf_contact_forces_z[0:i])
            return [rf_contact_forces_x_ani, rf_contact_forces_y_ani, rf_contact_forces_z_ani, total_weight_line, ax]
        
        rf_contact_forces_x = self.states_dict['rf_contact_forces_x']
        rf_contact_forces_y = self.states_dict['rf_contact_forces_y']
        rf_contact_forces_z = self.states_dict['rf_contact_forces_z']

        ax.set_xlim(0, self.episode_length)
        ax.set_ylim(-50, 1000)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('contact forces (N)')
        tick_loc = np.arange(0, self.episode_length, 100)
        tick_labels = tick_loc / 100
        ax.xaxis.set_ticks(tick_loc)
        ax.xaxis.set_ticklabels(tick_labels)
        ax.grid(ls='--')

        rf_contact_forces_x_ani, = ax.plot([], [], color='r', label='right foot contact forces x')
        rf_contact_forces_y_ani, = ax.plot([], [], color='g', label='right foot contact forces y')
        rf_contact_forces_z_ani, = ax.plot([], [], color='b', label='right foot contact forces z')
        total_weight_line = ax.axhline(y=self.env.total_weight, color='k', linestyle='--', label='total weight')

        ax.legend(loc='upper right')

        # * Define the animation for left foot contact forces
        bx = fig.add_subplot(spec[1])

        def lf_contact_forces_2D_init():
            lf_contact_forces_x_ani.set_data(np.linspace(0, 1, 0), lf_contact_forces_x[0:0])
            lf_contact_forces_y_ani.set_data(np.linspace(0, 1, 0), lf_contact_forces_y[0:0])
            lf_contact_forces_z_ani.set_data(np.linspace(0, 1, 0), lf_contact_forces_z[0:0])
            return [lf_contact_forces_x_ani, lf_contact_forces_y_ani, lf_contact_forces_z_ani, total_weight_line]

        def lf_contact_forces_2D_update(i):
            lf_contact_forces_x_ani.set_data(np.linspace(0, i-1, i), lf_contact_forces_x[0:i])
            lf_contact_forces_y_ani.set_data(np.linspace(0, i-1, i), lf_contact_forces_y[0:i])
            lf_contact_forces_z_ani.set_data(np.linspace(0, i-1, i), lf_contact_forces_z[0:i])
            return [lf_contact_forces_x_ani, lf_contact_forces_y_ani, lf_contact_forces_z_ani, total_weight_line, bx]
        
        lf_contact_forces_x = self.states_dict['lf_contact_forces_x']
        lf_contact_forces_y = self.states_dict['lf_contact_forces_y']
        lf_contact_forces_z = self.states_dict['lf_contact_forces_z']

        bx.set_xlim(0, self.episode_length)
        bx.set_ylim(-50, 1000)
        bx.set_xlabel('time (s)')
        bx.set_ylabel('contact forces (N)')
        tick_loc = np.arange(0, self.episode_length, 100)
        tick_labels = tick_loc / 100
        bx.xaxis.set_ticks(tick_loc)
        bx.xaxis.set_ticklabels(tick_labels)
        bx.grid(ls='--')

        lf_contact_forces_x_ani, = bx.plot([], [], color='r', label='left foot contact forces x')
        lf_contact_forces_y_ani, = bx.plot([], [], color='g', label='left foot contact forces y')
        lf_contact_forces_z_ani, = bx.plot([], [], color='b', label='left foot contact forces z')
        total_weight_line = bx.axhline(y=self.env.total_weight, color='k', linestyle='--', label='total weight')
        bx.legend(loc='upper right')

        # * Combine all the animations
        def _init_func():
            artist1 = rf_contact_forces_2D_init()
            artist2 = lf_contact_forces_2D_init()
            return artist1 + artist2
        
        def _update_func(i):
            artist1 = rf_contact_forces_2D_update(i)
            artist2 = lf_contact_forces_2D_update(i)
            return artist1 + artist2
        
        anim = FuncAnimation(fig=fig, init_func=_init_func, func=_update_func, frames=range(self.episode_length), interval=50, blit=False)

        # * Save the animation
        filepath = os.path.join(self.folderpath, f"{self.run_name}_contact_forces.mp4")
        anim.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])

        print(f"Contact forces animation saved to {filepath}")

    def make_video(self):
        print("Creating video...")
        filepath = os.path.join(self.folderpath, f"{self.run_name}_gym.mp4")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (self.frames[0].shape[1], self.frames[0].shape[0]))

        # Write the frames to the video file
        for frame in self.frames:
            cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(cv_frame)

        # Release the video writer and print completion message
        out.release()
        print(f"Video saved to {filepath}")

class CSVLogger():
    """ Log states to csv files """
    def __init__(self, env, experiment_name, log_dir, max_it):
        self.env = env
        self.experiment_name = experiment_name
        self.run_name = os.path.basename(log_dir)
        self.info_array = np.zeros((max_it, 1+13+20)) # Time, root_states, joint pos/vel
        self.episode_length = 0
        self.headers = ["ts",
                        "bp_x", "bp_y", "bp_z", 
                        "bq_w", "bq_x", "bq_y", "bq_z",
                        "bv_x", "bv_y", "bv_z", 
                        "bw_x", "bw_y", "bw_z",
                        "rj0_p", "rj1_p", "rj2_p", "rj3_p", "rj4_p",
                        "lj0_p", "lj1_p", "lj2_p", "lj3_p", "lj4_p",
                        "rj0_v", "rj1_v", "rj2_v", "rj3_v", "rj4_v",
                        "lj0_v", "lj1_v", "lj2_v", "lj3_v", "lj4_v",]

    def log(self):
        # Log time
        self.info_array[self.episode_length, 0] = self.episode_length*self.env.dt

        # Log root_states
        self.info_array[self.episode_length, 1:14] = self.env.root_states[0].cpu().numpy()

        # Re-shuffle [x,y,z,w] to [w,x,y,z]
        self.info_array[self.episode_length, 4] = self.env.root_states[0, 6].cpu().numpy()
        self.info_array[self.episode_length, 5:8] = self.env.root_states[0, 3:6].cpu().numpy()

        # Log joint pos/vel
        self.info_array[self.episode_length, 14:24] = self.env.dof_pos[0].cpu().numpy()
        self.info_array[self.episode_length, 24:34] = self.env.dof_vel[0].cpu().numpy()

        self.episode_length += 1
        
    def save_and_exit(self):
        folderpath = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.experiment_name, 'analysis')
        os.makedirs(folderpath, exist_ok=True)
        self.save_csv(folderpath)
        exit()

    def save_csv(self, folderpath):
        print("Saving csv...")
        filepath = os.path.join(folderpath, f"{self.run_name}.csv")
        np.savetxt(filepath, self.info_array[:self.episode_length, :], delimiter=",", header=",".join(self.headers), comments='')
        print(f"CSV saved to {filepath}")

class DictLogger():
    """ Log states/commands to npz files """
    def __init__(self, env, experiment_name, log_dir, max_it):
        self.env = env
        self.env_idx = 0
        self.experiment_name = experiment_name
        self.run_name = os.path.basename(log_dir)
        self.episode_length = 0
        self.data_dict = {'episode_length': self.episode_length}

        self.potential_keys = ["root_states", "dof_pos", "dof_vel", "commands",
                               "foot_states", "foot_heading", "foot_contact", "phase", "current_step",
                               "step_commands", "CoM", "LIPM_CoM"]
        self.keys = []
        
        for key in self.potential_keys:
            try:
                # Attempt to retrieve the attribute
                attr_value = getattr(self.env, key)
                # If successful, initialize the np arrays in the data dictionary and append the key
                self.data_dict[key] = np.zeros((max_it, *(attr_value.shape[1:])))
                self.keys.append(key)
            except AttributeError:
                # If the attribute doesn't exist, skip to the next key
                continue

    def log(self):
        for key in self.keys:
            self.data_dict[key][self.episode_length] = getattr(self.env, key)[self.env_idx].cpu().numpy()

        self.episode_length += 1
        self.data_dict['episode_length'] = self.episode_length
        
    def save_and_exit(self):
        folderpath = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.experiment_name, 'analysis')
        os.makedirs(folderpath, exist_ok=True)
        self.save_dict(folderpath)
        exit()

    def save_dict(self, folderpath):
        print("Saving list of dictionaries to npz...")
        filepath = os.path.join(folderpath, f"{self.run_name}.npz")
        np.savez(filepath, **self.data_dict)
        print(f"Dictionary saved to {filepath}")

class SuccessRater():
    def __init__(self, env, experiment_name, log_dir):
        self.env = env
        self.experiment_name = experiment_name
        self.run_name = os.path.basename(log_dir)
        self.folderpath = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.experiment_name, 'analysis')
        os.makedirs(self.folderpath, exist_ok=True)
        self.success_rate = 0.
        self.reset_cnt = 0
        self.timed_out_cnt = 0
        self.keys = ["0.0m/s", "0.1m/s", "0.2m/s", "0.3m/s", "0.4m/s", "0.5m/s", 
                               "0.6m/s", "0.7m/s", "0.8m/s", "0.9m/s", "1.0m/s",]
                            #    "1.1m/s", "1.2m/s", "1.3m/s", "1.4m/s", "1.5m/s",]

        self.data_dict = {}
        for key in self.keys:
            self.data_dict[key] = {'reset_cnt': 0, 'timed_out_cnt': 0, 'success_rate': 0.}

    def log(self, i, test_episodes, reset_cnt, timed_out_cnt):
        key_idx = i // test_episodes
        self.data_dict[self.keys[key_idx]]['reset_cnt'] += reset_cnt
        self.data_dict[self.keys[key_idx]]['timed_out_cnt'] += timed_out_cnt

    def save_and_exit(self):
        print("Calculating success rate...")
        for key in self.keys:
            self.data_dict[key]['success_rate'] = self.data_dict[key]['timed_out_cnt'] / self.data_dict[key]['reset_cnt']
        
        print("Saving list of dictionaries to npz...")
        filepath = os.path.join(self.folderpath, f"{self.run_name}.npz")
        np.savez(filepath, **self.data_dict)
        print(f"Dictionary saved to {filepath}")
        exit()
