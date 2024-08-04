import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from gym import LEGGED_GYM_ROOT_DIR
import os 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

font = {'size': 16}
mpl.rc('font', **font)
line = {'linewidth': 2}
mpl.rc('lines', **line)

COLORS = ['tab:pink', 'tab:cyan']
CMD_COLORS = ['red', 'blue']

def load_npz_data(e2e_flat_filepath, 
                  e2e_rough_filepath, 
                  stepper_filepath, 
                  stepper_gap_filepath,
                  raibert_filepath,
                  e2e_rough_sr_filepath,
                  e2e_gap_sr_filepath,
                  stepper_rough_sr_filepath,
                  stepper_gap_sr_filepath,
                  raibert_rough_sr_filepath):
                  
    # * Load the .npz file
    e2e_flat_data = np.load(e2e_flat_filepath, allow_pickle=True)
    e2e_rough_data = np.load(e2e_rough_filepath, allow_pickle=True)
    stepper_data = np.load(stepper_filepath, allow_pickle=True) 
    stepper_gap_data = np.load(stepper_gap_filepath, allow_pickle=True)
    raibert_data = np.load(raibert_filepath, allow_pickle=True)
    e2e_rough_sr_data = np.load(e2e_rough_sr_filepath, allow_pickle=True)
    e2e_gap_sr_data = np.load(e2e_gap_sr_filepath, allow_pickle=True)
    stepper_rough_sr_data = np.load(stepper_rough_sr_filepath, allow_pickle=True)
    stepper_gap_sr_data = np.load(stepper_gap_sr_filepath, allow_pickle=True)
    raibert_rough_sr_data = np.load(raibert_rough_sr_filepath, allow_pickle=True)

    return e2e_flat_data, e2e_rough_data, stepper_data, stepper_gap_data, raibert_data, \
           e2e_rough_sr_data, e2e_gap_sr_data, stepper_rough_sr_data, stepper_gap_sr_data, raibert_rough_sr_data

def plot_velocity_comparison(e2e_flat_data, e2e_rough_data, stepper_data, raibert_data, start_idx, end_idx):

    # Set x-ticks to represent seconds
    episode_length = 100  # 100 episodes correspond to 1 second
    tick_interval = 5 * episode_length  # For ticks every 0.5 seconds, adjust as needed
    ticks = np.arange(start_idx, end_idx, tick_interval)
    tick_labels = [f"{(i-start_idx)/episode_length:.0f}" for i in ticks]  # Convert episode length to seconds

    ### * Plot v_x velocity comparison * ###
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    ax.plot(np.arange(start_idx, end_idx), e2e_flat_data['commands'][start_idx:end_idx, 0], label=r"Reference", color='black', linestyle='--')

    ax.plot(np.arange(start_idx, end_idx), raibert_data['root_states'][start_idx:end_idx, 7], label=r"Raibert Heuristic", color='#EB984E')
    ax.plot(np.arange(start_idx, end_idx), e2e_flat_data['root_states'][start_idx:end_idx, 7], label=r"End-to-End flat", color='#1ABC9C')
    ax.plot(np.arange(start_idx, end_idx), e2e_rough_data['root_states'][start_idx:end_idx, 7], label=r"End-to-End mixed", color='#F4D03F')#5AFF54
    ax.plot(np.arange(start_idx, end_idx), stepper_data['root_states'][start_idx:end_idx, 7], label=r"Our Method", color='#1F618D')#0092FF
    ax.set_xlabel('Episode Length')
    ax.set_ylabel(r'Velocity [$m/s$]')
    # ax.set_title('X-Velocity Comparison')
    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Define the desired order of your handles and labels, by indices
    new_order = [0, 4, 2, 3, 1]  # for example, to put the fourth item first, etc.

    # Reorder handles and labels
    ordered_handles = [handles[idx] for idx in new_order]
    ordered_labels = [labels[idx] for idx in new_order]

    # Create the new legend with the reordered handles and labels
    ax.legend(ordered_handles, ordered_labels, loc='upper left', fontsize=15)

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Time [s]')  # Adjust label to indicate unit is now seconds

    # Save figure to pdf format
    fig.savefig(f'{LEGGED_GYM_ROOT_DIR}/logs/test/vx_comparison.pdf', format='pdf', bbox_inches='tight')
    
    ### * Plot v_y velocity comparison * ###
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(np.arange(start_idx, end_idx), e2e_flat_data['commands'][start_idx:end_idx, 1], label=r"Reference $\hat{v}_{y}$", color='black', linestyle='--')
    ax.plot(np.arange(start_idx, end_idx), stepper_data['root_states'][start_idx:end_idx, 8], label=r"Ours $v_{y}$")
    ax.plot(np.arange(start_idx, end_idx), e2e_flat_data['root_states'][start_idx:end_idx, 8], label=r"End-to-End flat $v_{y}$")
    ax.plot(np.arange(start_idx, end_idx), e2e_rough_data['root_states'][start_idx:end_idx, 8], label=r"End-to-End rough $v_{y}$")
    ax.set_xlabel('Episode Length')
    ax.set_ylabel(r'Velocity [$m/s$]')
    # ax.set_title('Y-Velocity Comparison')
    ax.legend()

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Time [s]')  # Adjust label to indicate unit is now seconds

    # Save figure to pdf format
    fig.savefig(f'{LEGGED_GYM_ROOT_DIR}/logs/test/vy_comparison.pdf', format='pdf', bbox_inches='tight')


def plot_gait_comparison(e2e_flat_data, stepper_data, start_idx, end_idx):
    ### * Plot E2E gait sequence comparison * ###
    fig, ax = plt.subplots(figsize=(10, 2))

    # Plot the ground truth gait sequence
    gt_gait_data = np.array([e2e_flat_data['foot_contact'][start_idx:end_idx,0], 
                             e2e_flat_data['foot_contact'][start_idx:end_idx,1]])
    
    leg_names = ['Right', 'Left']

    for idx, (leg_data, color) in enumerate(zip(gt_gait_data, COLORS)):
        # Find the start points and durations of the stance phases
        stance_starts = np.where(np.diff(leg_data, prepend=0, append=0) == 1)[0]
        stance_ends = np.where(np.diff(leg_data, prepend=0, append=0) == -1)[0]

        durations = stance_ends - stance_starts
        
        # Create a series of broken horizontal bars for each leg
        bars = [(start, duration) for start, duration in zip(stance_starts, durations)]
        ax.broken_barh(bars, (idx - 0.4, 0.8), facecolors=color, alpha=0.5, label='Actual')

    # Plot the actual gait sequence
    gait_data = np.array([np.array(e2e_flat_data['phase'][start_idx:end_idx,0] < 0.5, dtype=int), 
                          np.array(e2e_flat_data['phase'][start_idx:end_idx,0] >= 0.5, dtype=int)])

    for idx, (leg_data, color) in enumerate(zip(gait_data, CMD_COLORS)):
        # Find the start points and durations of the stance phases
        stance_starts = np.where(np.diff(leg_data, prepend=0, append=0) == 1)[0]
        stance_ends = np.where(np.diff(leg_data, prepend=0, append=0) == -1)[0]

        durations = stance_ends - stance_starts
        
        # Create a series of broken horizontal bars for each leg
        bars = [(start, duration) for start, duration in zip(stance_starts, durations)]
        ax.broken_barh(bars, (idx - 0.4, 0.8), facecolors=color, alpha=0.5, label='Ground Truth')

    # Set the y-ticks to be in the middle of each leg's bar
    ax.set_yticks(np.arange(gt_gait_data.shape[0]))
    ax.set_yticklabels(['Right', 'Left'])

    # Set the limits and labels
    ax.set_ylim(-1, gt_gait_data.shape[0])
    ax.set_xlim(0, gt_gait_data.shape[1])
    ax.set_xlabel('Episode Length')
    ax.set_title('End-to-End Gait Sequence')
    ax.legend()

    # Remove the spines on the top and right sides
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.savefig(f'{LEGGED_GYM_ROOT_DIR}/logs/test/e2e_gait_sequence.pdf', format='pdf')


    ### * Plot Stepper gait sequence comparison * ###
    fig, ax = plt.subplots(figsize=(10, 2))

    # Plot the ground truth gait sequence
    gt_gait_data = np.array([stepper_data['foot_contact'][start_idx:end_idx,0], 
                             stepper_data['foot_contact'][start_idx:end_idx,1]])
    
    for idx, (leg_data, color) in enumerate(zip(gt_gait_data, COLORS)):
        # Find the start points and durations of the stance phases
        stance_starts = np.where(np.diff(leg_data, prepend=0, append=0) == 1)[0]
        stance_ends = np.where(np.diff(leg_data, prepend=0, append=0) == -1)[0]

        durations = stance_ends - stance_starts
        
        # Create a series of broken horizontal bars for each leg
        bars = [(start, duration) for start, duration in zip(stance_starts, durations)]
        ax.broken_barh(bars, (idx - 0.4, 0.8), facecolors=color, alpha=0.6, label=f'Measured {leg_names[idx]} Step')

    # Plot the actual gait sequence
    gait_data = np.array([np.array(stepper_data['phase'][start_idx:end_idx,0] < 0.5, dtype=int), 
                          np.array(stepper_data['phase'][start_idx:end_idx,0] >= 0.5, dtype=int)])

    for idx, (leg_data, color) in enumerate(zip(gait_data, CMD_COLORS)):
        # Find the start points and durations of the stance phases
        stance_starts = np.where(np.diff(leg_data, prepend=0, append=0) == 1)[0]
        stance_ends = np.where(np.diff(leg_data, prepend=0, append=0) == -1)[0]

        durations = stance_ends - stance_starts
        
        # Create a series of broken horizontal bars for each leg
        bars = [(start, duration) for start, duration in zip(stance_starts, durations)]
        ax.broken_barh(bars, (idx - 0.4, 0.8), facecolors=color, alpha=0.6, label=f'Desired {leg_names[idx]} Step')

    # Set the y-ticks to be in the middle of each leg's bar
    ax.set_yticks(np.arange(gt_gait_data.shape[0]))
    ax.set_yticklabels(['Right', 'Left'])

    # Set the limits and labels
    ax.set_ylim(-0.6, gt_gait_data.shape[0]-0.5)
    ax.set_xlim(0, gt_gait_data.shape[1])
    # ax.set_title('Stepper Gait Sequence')

    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Define the desired order of your handles and labels, by indices
    new_order = [1, 0, 3, 2]  # for example, to put the fourth item first, etc.

    # Reorder handles and labels
    ordered_handles = [handles[idx] for idx in new_order]
    ordered_labels = [labels[idx] for idx in new_order]

    # Create the new legend with the reordered handles and labels
    ax.legend(ordered_handles, ordered_labels, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.6))
    # ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.6))

    # Set x-ticks to represent seconds
    episode_length = 100  # 100 episodes correspond to 1 second
    tick_interval = 1 * episode_length  # For ticks every 0.5 seconds, adjust as needed
    ticks = np.arange(start_idx, end_idx, tick_interval) - start_idx
    tick_labels = [f"{(i+start_idx)/episode_length:.0f}" for i in ticks]  # Convert episode length to seconds

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(r'Time [$s$]')  # Adjust label to indicate unit is now seconds

    # Remove the spines on the top and right sides
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.savefig(f'{LEGGED_GYM_ROOT_DIR}/logs/test/stepper_gait_sequence.pdf', format='pdf', bbox_inches='tight')


def plot_3d_foot_trajectory(e2e_flat_data, stepper_data, start_idx, end_idx):
    ### * Plot E2E 3D foot trajectory * ###
    right_x = e2e_flat_data['foot_states'][start_idx:end_idx, 0,0]
    right_y = e2e_flat_data['foot_states'][start_idx:end_idx, 0,1]
    right_z = e2e_flat_data['foot_states'][start_idx:end_idx, 0,2]

    left_x = e2e_flat_data['foot_states'][start_idx:end_idx, 1,0]
    left_y = e2e_flat_data['foot_states'][start_idx:end_idx, 1,1]
    left_z = e2e_flat_data['foot_states'][start_idx:end_idx, 1,2]

    base_x = e2e_flat_data['root_states'][start_idx:end_idx, 0]
    base_y = e2e_flat_data['root_states'][start_idx:end_idx, 1]
    base_z = e2e_flat_data['root_states'][start_idx:end_idx, 2]

    e2e_CoM_x = e2e_flat_data['CoM'][start_idx:end_idx, 0]
    e2e_CoM_y = e2e_flat_data['CoM'][start_idx:end_idx, 1]
    e2e_CoM_z = e2e_flat_data['CoM'][start_idx:end_idx, 2]

    # Setting up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   
    # Plotting the trajectory
    ax.plot(right_x, right_y, right_z, label='Right foot trajectory', color=COLORS[0])
    ax.plot(left_x, left_y, left_z, label='Left foot trajectory', color=COLORS[1])
    ax.plot(base_x, base_y, base_z, label='Base trajectory', color='black')
    ax.plot(e2e_CoM_x, e2e_CoM_y, e2e_CoM_z, label='CoM trajectory', color='green')

    # Labeling the axes
    ax.set_xlabel(r'Forward Displacement [$m$]')
    ax.set_ylabel(r'Lateral Displacement [$m$]')
    ax.set_zlabel(r'Vertical Displacement [$m$]')
    ax.set_title('End-to-End 3D Foot Trajectory')

    # Adding a legend
    ax.legend()

    # Setting the viewing angle for better visualization
    ax.view_init(elev=20., azim=-35)
    fig.savefig(f'{LEGGED_GYM_ROOT_DIR}/logs/test/e2e_foot_trajectory.pdf', format='pdf')

    ### * Plot Stepper 3D foot trajectory * ###
    right_x = stepper_data['foot_states'][start_idx:end_idx, 0,0]
    right_y = stepper_data['foot_states'][start_idx:end_idx, 0,1]
    right_z = stepper_data['foot_states'][start_idx:end_idx, 0,2]

    left_x = stepper_data['foot_states'][start_idx:end_idx, 1,0]
    left_y = stepper_data['foot_states'][start_idx:end_idx, 1,1]
    left_z = stepper_data['foot_states'][start_idx:end_idx, 1,2]

    right_cmd_x = stepper_data['step_commands'][start_idx:end_idx, 0,0]
    right_cmd_y = stepper_data['step_commands'][start_idx:end_idx, 0,1]
    right_cmd_z = np.zeros_like(right_cmd_x)

    left_cmd_x = stepper_data['step_commands'][start_idx:end_idx, 1,0]
    left_cmd_y = stepper_data['step_commands'][start_idx:end_idx, 1,1]
    left_cmd_z = np.zeros_like(left_cmd_x)

    base_x = stepper_data['root_states'][start_idx:end_idx, 0]
    base_y = stepper_data['root_states'][start_idx:end_idx, 1]
    base_z = stepper_data['root_states'][start_idx:end_idx, 2]

    CoM_x = stepper_data['CoM'][start_idx:end_idx, 0]
    CoM_y = stepper_data['CoM'][start_idx:end_idx, 1]
    CoM_z = stepper_data['CoM'][start_idx:end_idx, 2]

    LIPM_CoM_x = stepper_data['LIPM_CoM'][start_idx:end_idx, 0]
    LIPM_CoM_y = stepper_data['LIPM_CoM'][start_idx:end_idx, 1]
    LIPM_CoM_z = stepper_data['LIPM_CoM'][start_idx:end_idx, 2]

    # Setting up the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get rid of the panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.set_xlim(1.2, 2.8)
    ax.set_ylim(0.2, 0.45)
    ax.set_zlim(-0.01, 0.65)
    # ax3d.set_aspect('auto')
    ax.set_box_aspect([1.6,1.5,0.66])

    # Plotting the trajectory
    ax.plot(LIPM_CoM_x, LIPM_CoM_y, LIPM_CoM_z, label='LIPM CoM Trajectory', color='g', linestyle='--')#, alpha=0.7)
    ax.plot(left_x, left_y, left_z, label='Left Foot Trajectory', color=COLORS[1])
    ax.scatter(left_cmd_x, left_cmd_y, left_cmd_z, label='Desired Left Step', color=CMD_COLORS[1], marker='x', s=30)
    ax.plot(CoM_x, CoM_y, CoM_z, label='Measured CoM Trajectory', color='#1F618D')
    ax.plot(right_x, right_y, right_z, label='Right Foot Trajectory', color=COLORS[0])
    ax.scatter(right_cmd_x, right_cmd_y, right_cmd_z, label='Desired Right Step', color=CMD_COLORS[0], marker='x', s=30)
    # ax.plot(base_x, base_y, base_z, label='Base trajectory', color='black')

    # Labeling the axes
    ax.set_xlabel(r'x [$m$]')
    ax.set_ylabel(r'y [$m$]')
    ax.set_zlabel(r'z [$m$]')
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    # ax.zaxis.labelpad = 10
    # ax.set_title('Stepper 3D Foot Trajectory')

    # Adding a legend
    ax.legend(ncol=2, loc='upper center',)

    # Setting the viewing angle for better visualization
    ax.view_init(elev=25., azim=255)
    fig.savefig(f'{LEGGED_GYM_ROOT_DIR}/logs/test/stepper_foot_trajectory.pdf', format='pdf')

def plot_gap_crossing(stepper_gap_data, start_idx, end_idx):

    ### * Plot Stepper gap crossing * ###
    right_x = stepper_gap_data['foot_states'][start_idx:end_idx, 0,0]
    right_y = stepper_gap_data['foot_states'][start_idx:end_idx, 0,1]
    right_z = stepper_gap_data['foot_states'][start_idx:end_idx, 0,2]

    left_x = stepper_gap_data['foot_states'][start_idx:end_idx, 1,0]
    left_y = stepper_gap_data['foot_states'][start_idx:end_idx, 1,1]
    left_z = stepper_gap_data['foot_states'][start_idx:end_idx, 1,2]

    right_cmd_x = stepper_gap_data['step_commands'][start_idx:end_idx, 0,0]
    right_cmd_y = stepper_gap_data['step_commands'][start_idx:end_idx, 0,1]
    right_cmd_z = np.zeros_like(right_cmd_x)

    left_cmd_x = stepper_gap_data['step_commands'][start_idx:end_idx, 1,0]
    left_cmd_y = stepper_gap_data['step_commands'][start_idx:end_idx, 1,1]
    left_cmd_z = np.zeros_like(left_cmd_x)

    base_x = stepper_gap_data['root_states'][start_idx:end_idx, 0]
    base_y = stepper_gap_data['root_states'][start_idx:end_idx, 1]
    base_z = stepper_gap_data['root_states'][start_idx:end_idx, 2]

    CoM_x = stepper_gap_data['CoM'][start_idx:end_idx, 0]
    CoM_y = stepper_gap_data['CoM'][start_idx:end_idx, 1]
    CoM_z = stepper_gap_data['CoM'][start_idx:end_idx, 2]

    LIPM_CoM_x = stepper_gap_data['LIPM_CoM'][start_idx:end_idx, 0]
    LIPM_CoM_y = stepper_gap_data['LIPM_CoM'][start_idx:end_idx, 1]
    LIPM_CoM_z = stepper_gap_data['LIPM_CoM'][start_idx:end_idx, 2]

    # Setting up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the trajectory
    ax.plot(right_x, right_y, right_z, label='Right foot trajectory', color=COLORS[0])
    ax.plot(left_x, left_y, left_z, label='Left foot trajectory', color=COLORS[1])
    ax.scatter(right_cmd_x, right_cmd_y, right_cmd_z, label='Right foot command', color=CMD_COLORS[0], s=5)
    ax.scatter(left_cmd_x, left_cmd_y, left_cmd_z, label='Left foot command', color=CMD_COLORS[1], s=5)
    # ax.plot(base_x, base_y, base_z, label='Base trajectory', color='black')
    ax.plot(CoM_x, CoM_y, CoM_z, label='CoM trajectory', color='green')
    ax.plot(LIPM_CoM_x, LIPM_CoM_y, LIPM_CoM_z, label='LIPM CoM trajectory', color='orange')
    
    # Labeling the axes
    ax.set_xlabel('Forward Displacement (m)')
    ax.set_ylabel('Lateral Displacement (m)')
    ax.set_zlabel('Vertical Displacement (m)')
    ax.set_title('Stepper Gap Crossing Foot Trajectory')

    # Adding a legend
    ax.legend()

    # Setting the viewing angle for better visualization
    ax.view_init(elev=20., azim=-35)
    fig.savefig(f'{LEGGED_GYM_ROOT_DIR}/logs/test/stepper_gap_crossing.pdf', format='pdf')


def plot_success_rate(e2e_rough_sr_data, stepper_rough_sr_data, raibert_rough_sr_data):
    ### * Plot Success Rate * ###
    fig, ax = plt.subplots(figsize=(10, 6))

    e2e_success_rate = []
    stepper_success_rate = []
    raibert_success_rate = []
    keys = e2e_rough_sr_data.files
    xticklabels = []

    bar_width = 0.35 / 3 * 2 
    x = np.arange(len(keys))

    for key in keys:
        e2e_success_rate.append(e2e_rough_sr_data[key].item()['success_rate'])
        stepper_success_rate.append(stepper_rough_sr_data[key].item()['success_rate'])
        raibert_success_rate.append(raibert_rough_sr_data[key].item()['success_rate'])
        xticklabels.append(key[:3])

    ax.bar(x - bar_width, stepper_success_rate, bar_width, label='Our Method', color='#1F618D')
    ax.bar(x            , e2e_success_rate, bar_width, label='End-to-End', color='#1ABC9C')
    ax.bar(x + bar_width, raibert_success_rate, bar_width, label='Raibert Heuristic', color='#EB984E')
    
    ax.set_xlabel(r'Forward Command Velocity $\hat{v}_x$ [$m/s$]')
    ax.set_ylabel('Success Rate')
    # ax.set_title('Rough Terrain Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.13))

    fig.savefig(f'{LEGGED_GYM_ROOT_DIR}/logs/test/Rough_success_rate_old.pdf', format='pdf', bbox_inches='tight')

    ### * Plot Success Rate2 * ###
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 3.5), gridspec_kw={'height_ratios': [2, 1]})

    # Plot on the first (top) subplot
    ax1.bar(x - bar_width, stepper_success_rate, bar_width, label='Our Method', color='#1F618D')
    ax1.bar(x            , e2e_success_rate, bar_width, label='End-to-End', color='#1ABC9C')
    ax1.bar(x + bar_width, raibert_success_rate, bar_width, label='Raibert Heuristic', color='#EB984E')

    # Plot on the second (bottom) subplot
    ax2.bar(x - bar_width, stepper_success_rate, bar_width, color='#1F618D')
    ax2.bar(x            , e2e_success_rate, bar_width, color='#1ABC9C')
    ax2.bar(x + bar_width, raibert_success_rate, bar_width, color='#EB984E')
    
    # Set the y-axis limits to create the break effect
    ax1.set_ylim(0.75, 1.0)  # upper range of the success rate
    ax2.set_ylim(0.0, 0.6)   # lower range of the success rate

    # Hide the spines between ax1 and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(top=False, bottom=False)
    ax2.xaxis.tick_bottom()

    # Diagonal lines to indicate the break in the y-axis
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1.03 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1.03 + d), **kwargs)  # bottom-right diagonal

    # Labels and legend
    ax2.set_xlabel(r'Forward Command Velocity $\hat{v}_x$ [$m/s$]')
    ax2.set_ylabel('Success Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(xticklabels)
    # Adjust y-axis label position
    ax2.yaxis.set_label_coords(-0.06, 1.5)
    ax1.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.4))

    # Save the figure
    fig.savefig(f'{LEGGED_GYM_ROOT_DIR}/logs/test/Rough_success_rate.pdf', format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'test')
                             
    e2e_flat_filepath = os.path.join(root_path, 'e2e_flat.npz')
    e2e_rough_filepath = os.path.join(root_path, 'e2e_rough.npz')
    stepper_filepath = os.path.join(root_path, 'stepper.npz')
    stepper_gap_filepath = os.path.join(root_path, 'stepper_gap.npz')
    raibert_filepath = os.path.join(root_path, 'raibert.npz')
    e2e_rough_sr_filepath = os.path.join(root_path, 'e2e_rough_sr9.npz')
    e2e_gap_sr_filepath = os.path.join(root_path, 'e2e_gap_sr.npz')
    stepper_rough_sr_filepath = os.path.join(root_path, 'stepper_rough_sr9.npz')
    stepper_gap_sr_filepath = os.path.join(root_path, 'stepper_gap_sr.npz')
    raibert_rough_sr_filepath = os.path.join(root_path, 'raibert_rough_sr2.npz')
                                
    e2e_flat_data, e2e_rough_data, \
    stepper_data, stepper_gap_data, \
    raibert_data, \
    e2e_rough_sr_data, e2e_gap_sr_data, \
    stepper_rough_sr_data, stepper_gap_sr_data, \
    raibert_rough_sr_data = load_npz_data(e2e_flat_filepath,
                                        e2e_rough_filepath,
                                        stepper_filepath,
                                        stepper_gap_filepath,
                                        raibert_filepath,
                                        e2e_rough_sr_filepath,
                                        e2e_gap_sr_filepath,
                                        stepper_rough_sr_filepath,
                                        stepper_gap_sr_filepath,
                                        raibert_rough_sr_filepath)
    
    episode_length = e2e_flat_data['episode_length'].item()
    start, end = 0, episode_length
    # start, end = 2200, 2700
    start, end = 730, 1010

    # vel_start, vel_end = 0, episode_length
    # plot_velocity_comparison(e2e_flat_data, e2e_rough_data, stepper_data, raibert_data,
    #                          start_idx=vel_start, end_idx=vel_end) 
    
    # gait_start, gait_end = 700, 977
    # plot_gait_comparison(e2e_flat_data, stepper_data,
    #                     start_idx=gait_start, end_idx=gait_end)
    
    # ft_start, ft_end = 730, 1010
    # # ft_start, ft_end = 2400, 2700
    # plot_3d_foot_trajectory(e2e_flat_data, stepper_data,
    #                         start_idx=ft_start, end_idx=ft_end)

    # episode_length = stepper_gap_data['episode_length'].item()
    # start, end = 0, episode_length
    # plot_gap_crossing(stepper_gap_data, start_idx=start, end_idx=end)

    plot_success_rate(e2e_rough_sr_data, stepper_rough_sr_data, raibert_rough_sr_data)
    
    plt.show()
