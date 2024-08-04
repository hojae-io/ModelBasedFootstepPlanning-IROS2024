import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

import os
from LIPM_3D import LIPM3D
import matplotlib as mpl
from matplotlib.transforms import Bbox

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

font = {'size': 18}
mpl.rc('font', **font)
line = {'linewidth': 3}
mpl.rc('lines', **line)

class Ball:
    def __init__(self, size=10, color='blue', marker='o', alpha=1.0):
        self.scatter, = ax.plot([], [], [], color=color, marker=marker, markersize=size, alpha=alpha, animated=False)

    def update(self, pos):
        if None in pos:
            return
        # draw ball
        self.scatter.set_data_3d(pos)

class Marker:
    def __init__(self, size=10, color='blue', marker='x', alpha=1.0):
        self.scatter, = ax.plot([], [], [], color=color, marker=marker, markersize=size, alpha=alpha, animated=False)

    def update(self, pos):
        self.scatter.set_xdata(pos[0,:])
        self.scatter.set_ydata(pos[1,:])
        self.scatter.set_3d_properties(np.asarray(pos[2,:]))
        # if None in pos:
        #     return
        # # draw ball
        # self.scatter.set_data_3d(pos)

class Line:
    def __init__(self, size=1, color='g', alpha=1.0, linestyle='-'):
        self.line, = ax.plot([], [], [], linewidth=size, color=color, linestyle=linestyle, alpha=alpha, animated=False)

    def update(self, pos):
        # draw line
        self.line.set_xdata(pos[0,:])
        self.line.set_ydata(pos[1,:])
        self.line.set_3d_properties(np.asarray(pos[2,:]))
        
class Quadrant:
    def __init__(self, center, radius, start_u, end_u, start_v, end_v, color='gray'):
        self.color = color
        self.x, self.y, self.z = draw_quadrant(center, radius, start_u, end_u, start_v, end_v)
        self.surface = ax.plot_surface(self.x, self.y, self.z, color=self.color)

    def update(self, center, radius, start_u, end_u, start_v, end_v):
        self.x, self.y, self.z = draw_quadrant(center, radius, start_u, end_u, start_v, end_v)
        self.surface.remove()
        self.surface = ax.plot_surface(self.x, self.y, self.z, color=self.color)

class LIPM_3D_Animate():
    def __init__(self):
        self.origin = Ball(size=2, color='k', marker='o')
        self.COM_trajectory = Line(size=2, color='g')
        self.COM_trajectory_proj = Line(size=2, color='g', linestyle='--', alpha=0.8)
        self.eICP_trajcectory = Line(size=2, color='y', linestyle='--', alpha=0.8)
        self.COM_head = Ball(size=2, color='r', marker='o')

        self.left_foot = Ball(size=10, color='c', marker='o')
        self.right_foot = Ball(size=10, color='m', marker='o')

        self.left_step_command = Marker(size=10, color='b', marker='x')
        self.right_step_command = Marker(size=10, color='r', marker='x')

        self.left_leg = Line(size=2, color='k')
        self.right_leg = Line(size=2, color='k')
        self.COM = Ball(size=16, color='black', marker='o', alpha=0.7)

        self.q1 = Quadrant([0, 0, 0], 0.05, 0, np.pi/2, 0, np.pi/2, 'black')
        self.q2 = Quadrant([0, 0, 0], 0.05, 0, np.pi/2, np.pi/2, np.pi, 'white')
        self.q3 = Quadrant([0, 0, 0], 0.05, 0, np.pi/2, np.pi, 3*np.pi/2, 'black')
        self.q4 = Quadrant([0, 0, 0], 0.05, 0, np.pi/2, 3*np.pi/2, 2*np.pi, 'white')
        self.q5 = Quadrant([0, 0, 0], 0.05, np.pi/2, np.pi, 0, np.pi/2, 'white')
        self.q6 = Quadrant([0, 0, 0], 0.05, np.pi/2, np.pi, np.pi/2, np.pi, 'black')
        self.q7 = Quadrant([0, 0, 0], 0.05, np.pi/2, np.pi, np.pi, 3*np.pi/2, 'white')
        self.q8 = Quadrant([0, 0, 0], 0.05, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 'black')


    def update(self, COM_pos, COM_pos_trajectory, COM_pos_trajectory_proj, eICP_trajcectory, left_foot_pos, right_foot_pos, left_step_command, right_step_command):
        self.origin.update([0, 0, 0])
        self.COM.update(COM_pos)

        self.q1.update(COM_pos, 0.05, 0, np.pi/2, 0, np.pi/2)
        self.q2.update(COM_pos, 0.05, 0, np.pi/2, np.pi/2, np.pi)
        self.q3.update(COM_pos, 0.05, 0, np.pi/2, np.pi, 3*np.pi/2)
        self.q4.update(COM_pos, 0.05, 0, np.pi/2, 3*np.pi/2, 2*np.pi)
        self.q5.update(COM_pos, 0.05, np.pi/2, np.pi, 0, np.pi/2)
        self.q6.update(COM_pos, 0.05, np.pi/2, np.pi, np.pi/2, np.pi)
        self.q7.update(COM_pos, 0.05, np.pi/2, np.pi, np.pi, 3*np.pi/2)
        self.q8.update(COM_pos, 0.05, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi)

        self.COM_trajectory.update(COM_pos_trajectory)
        self.COM_trajectory_proj.update(COM_pos_trajectory_proj)
        self.eICP_trajcectory.update(eICP_trajcectory)
        self.COM_head.update(COM_pos_trajectory[:,-1])

        self.left_foot.update(left_foot_pos)
        self.right_foot.update(right_foot_pos)

        self.left_step_command.update(left_step_command)
        self.right_step_command.update(right_step_command)

        pos_1 = np.zeros((3,2))
        pos_1[:,0] = COM_pos
        pos_1[:,1] = left_foot_pos
        self.left_leg.update(pos_1)

        pos_2 = np.zeros((3,2))
        pos_2[:,0] = COM_pos
        pos_2[:,1] = right_foot_pos
        self.right_leg.update(pos_2)

        artists = []
        artists.append(self.origin.scatter)
        artists.append(self.left_leg.line)
        artists.append(self.right_leg.line)
        artists.append(self.COM.scatter)
        artists.append(self.COM_trajectory.line)
        artists.append(self.COM_head.scatter)
        artists.append(self.left_foot.scatter)
        artists.append(self.right_foot.scatter)
        artists.append(self.left_step_command.scatter)
        artists.append(self.right_step_command.scatter)

        artists.append(self.q1.surface)
        artists.append(self.q2.surface)
        artists.append(self.q3.surface)
        artists.append(self.q4.surface)
        artists.append(self.q5.surface)
        artists.append(self.q6.surface)
        artists.append(self.q7.surface)
        artists.append(self.q8.surface)

        # automatic set the x, y view limitation 
        if COM_pos[0] >= 3.0:
            ax.set_xlim(-1.0 + COM_pos[0] - 3.0, 4.0 + COM_pos[0] - 3.0)
        elif COM_pos[0] <= 0:
            ax.set_xlim(-1.0 + COM_pos[0], 4.0 + COM_pos[0])

        if COM_pos[1] >= 1.5:
            ax.set_ylim(-2 + COM_pos[1] - 1.5, 2 + COM_pos[1] - 1.5)
        elif COM_pos[1] <= -1.5:
            ax.set_ylim(-2 + COM_pos[1] + 1.5, 2 + COM_pos[1] + 1.5)

        ax.set_zlim(-0.01, 1.)

        ax.set_xlim(3.6, 5.0)
        ax.set_ylim(-1.0, -0.4)
        ax.set_zlim(-0.01, 0.65)

        artists.append(ax)

        return artists 

def ani_3D_init():
    return [] 

def ani_3D_update(i):
    COM_pos = [COM_pos_x[i], COM_pos_y[i], LIPM_model.zc]
    COM_pos_trajectory = np.zeros((3, i))
    COM_pos_trajectory[0,:] = COM_pos_x[0:i]
    COM_pos_trajectory[1,:] = COM_pos_y[0:i]
    COM_pos_trajectory[2,:] = LIPM_model.zc * np.ones((1,i))
    COM_pos_trajectory_proj = np.zeros((3, i))
    COM_pos_trajectory_proj[0,:] = COM_pos_x[0:i]
    COM_pos_trajectory_proj[1,:] = COM_pos_y[0:i]
    COM_pos_trajectory_proj[2,:] = np.zeros((1,i))

    eICP_trajcectory = np.zeros((3, i))
    eICP_trajcectory[0,:] = eICP_x[0:i]
    eICP_trajcectory[1,:] = eICP_y[0:i]
    eICP_trajcectory[2,:] = np.zeros((1,i))

    left_foot_pos = [left_foot_pos_x[i], left_foot_pos_y[i], left_foot_pos_z[i]]
    right_foot_pos = [right_foot_pos_x[i], right_foot_pos_y[i], right_foot_pos_z[i]]

    left_step_command = np.zeros((3,i))
    left_step_command[0,:] = left_step_command_x[0:i]
    left_step_command[1,:] = left_step_command_y[0:i]
    left_step_command[2,:] = np.zeros((1,i))

    right_step_command = np.zeros((3,i))
    right_step_command[0,:] = right_step_command_x[0:i]
    right_step_command[1,:] = right_step_command_y[0:i]
    right_step_command[2,:] = np.zeros((1,i))

    artists = LIPM_3D_ani.update(COM_pos, COM_pos_trajectory, COM_pos_trajectory_proj, eICP_trajcectory, left_foot_pos, right_foot_pos, left_step_command, right_step_command)

    return artists 

def ani_2D_init():
    COM_traj_ani.set_data(COM_pos_x[0:0], COM_pos_y[0:0])
    eICP_ani.set_data(eICP_x[0:0], eICP_y[0:0])
    COM_pos_ani.set_data(COM_pos_x[0], COM_pos_y[0])
    left_foot_pos_ani.set_data(left_foot_pos_x[0], left_foot_pos_y[0])
    right_foot_pos_ani.set_data(right_foot_pos_x[0], right_foot_pos_y[0])
    left_step_command_ani.set_data(left_step_command_x[0], left_step_command_y[0])
    right_step_command_ani.set_data(right_step_command_x[0], right_step_command_y[0])

    # Add the circle and wedges to the Axes
    COM_circle_ani.center = (COM_pos_x[0], COM_pos_y[0])
    COM_wedge_top_right_ani.set_center((COM_pos_x[0], COM_pos_y[0]))
    COM_wedge_bottom_left_ani.set_center((COM_pos_x[0], COM_pos_y[0]))

    return [COM_pos_ani, COM_traj_ani, left_foot_pos_ani, right_foot_pos_ani, left_step_command_ani, right_step_command_ani,
            COM_circle_ani, COM_wedge_top_right_ani, COM_wedge_bottom_left_ani, vel_cmd_ani]

def ani_2D_update(i):
    COM_traj_ani.set_data(COM_pos_x[0:i], COM_pos_y[0:i])
    eICP_ani.set_data(eICP_x[0:i], eICP_y[0:i])
    COM_pos_ani.set_data(COM_pos_x[i], COM_pos_y[i])
    left_foot_pos_ani.set_data(left_foot_pos_x[i], left_foot_pos_y[i])
    right_foot_pos_ani.set_data(right_foot_pos_x[i], right_foot_pos_y[i])
    left_step_command_ani.set_data(left_step_command_x[0:i], left_step_command_y[0:i])
    right_step_command_ani.set_data(right_step_command_x[0:i], right_step_command_y[0:i])

    # ani_text_COM_pos.set_text(COM_pos_str % (COM_pos_x[i], COM_pos_y[i]))

    # # automatic set the x, y view limitation 
    # bx.set_xlim(-2.0 + COM_pos_x[i], 3.0 + COM_pos_x[i])
    # bx.set_ylim(-0.8 + COM_pos_y[i], 0.8 + COM_pos_y[i])
    bx.set_xlim(3.6, 5.0)
    bx.set_ylim(-0.95, -0.45)

    # Update the position of the circle and wedges
    COM_circle_ani.center = (COM_pos_x[i], COM_pos_y[i])
    COM_wedge_top_right_ani.set_center((COM_pos_x[i], COM_pos_y[i]))
    COM_wedge_bottom_left_ani.set_center((COM_pos_x[i], COM_pos_y[i]))
    global vel_cmd_ani
    vel_cmd_ani.remove()  # Remove the old quiver
    vel_cmd_ani = bx.quiver(COM_pos_x[i], COM_pos_y[i], 0.15, 0, scale=1, color='#7FFF00') 

    return [COM_pos_ani, eICP_ani, COM_traj_ani, left_foot_pos_ani, right_foot_pos_ani, left_step_command_ani, right_step_command_ani, ani_text_COM_pos, bx,
            COM_circle_ani, COM_wedge_top_right_ani, COM_wedge_bottom_left_ani, vel_cmd_ani]
   

def COM_vel_2D_init():
    COM_vel_x_ani.set_data(np.linspace(0, 1, 0), COM_vel_x[0:0])
    COM_vel_y_ani.set_data(np.linspace(0, 1, 0), COM_vel_y[0:0])
    return [COM_vel_x_ani, COM_vel_y_ani]

def COM_vel_2D_update(i):
    COM_vel_x_ani.set_data(np.linspace(0, i-1, i), COM_vel_x[0:i])
    COM_vel_y_ani.set_data(np.linspace(0, i-1, i), COM_vel_y[0:i])
    COM_dvel_x_ani.set_data(np.linspace(0, i-1, i), COM_dvel_x[0:i])
    COM_dvel_y_ani.set_data(np.linspace(0, i-1, i), COM_dvel_y[0:i])
    return [COM_vel_x_ani, COM_vel_y_ani, cx]

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
    return [step_length_ani, step_width_ani, dx]


# %% ---------------------------------------------------------------- LIPM control
print('\n--------- Program start from here ...')

COM_pos_x = list()
COM_pos_y = list()
COM_vel_x = list()
COM_vel_y = list()
COM_dvel_x = list()
COM_dvel_y = list()
left_foot_pos_x = list()
left_foot_pos_y = list()
left_foot_pos_z = list()
right_foot_pos_x = list()
right_foot_pos_y = list()
right_foot_pos_z = list()
left_step_command_x = list()
left_step_command_y = list()
left_step_command_z = list()
right_step_command_x = list()
right_step_command_y = list()
right_step_command_z = list()
step_length = list()
dstep_length = list()
step_width = list() 
dstep_width = list()
eICP_x = list()
eICP_y = list()

# Initialize the 3D LIPM (with initial COM position, velocity and foot position)
# COM_pos_0 = [-0.4, 0.2, 1.0]
# COM_v0 = [1.0, -0.01]
COM_pos_0 = [0., 0., .6]
COM_v0 = [1.0, 0.]

left_foot_pos = [-0.2, 0.3, 0]
right_foot_pos = [-0.2, -0.3, 0]
support_foot_pos = np.array(left_foot_pos)
prev_support_foot_pos = np.array(left_foot_pos)

LIPM_model = LIPM3D(dt=0.02, T=0.35, s_d=0.6, w_d=0.3, support_leg='left_leg')
# LIPM_model = LIPM3D(dt=0.02, T=0.5, support_leg='right_leg')
LIPM_model.initializeModel(COM_pos_0, left_foot_pos, right_foot_pos)

LIPM_model.x_0 = LIPM_model.COM_pos[0] - LIPM_model.support_foot_pos[0] # origin is at the support foot
LIPM_model.y_0 = LIPM_model.COM_pos[1] - LIPM_model.support_foot_pos[1] 
LIPM_model.vx_0 = COM_v0[0]
LIPM_model.vy_0 = COM_v0[1]

LIPM_model.x_t = LIPM_model.x_0 
LIPM_model.y_t = LIPM_model.y_0
LIPM_model.vx_t = LIPM_model.vx_0
LIPM_model.vy_t = LIPM_model.vy_0

swing_data_len = int(LIPM_model.T/LIPM_model.dt)
swing_foot_pos = np.zeros((swing_data_len, 3))
j = 0

# Calculate the next step locations
LIPM_model.calculateFootLocationForNextStepXcoMWorld()
# LIPM_model.calculateFootLocationForNextStepXcoMBase()

# Calculate the foot positions for swing phase
if LIPM_model.support_leg == 'left_leg':
    right_foot_target_pos = [LIPM_model.u_x, LIPM_model.u_y, 0]
    swing_foot_pos[:,0] = np.linspace(LIPM_model.right_foot_pos[0], right_foot_target_pos[0], swing_data_len)
    swing_foot_pos[:,1] = np.linspace(LIPM_model.right_foot_pos[1], right_foot_target_pos[1], swing_data_len)
    swing_foot_pos[1:swing_data_len-1, 2] = 0.1
else:
    left_foot_target_pos = [LIPM_model.u_x, LIPM_model.u_y, 0]
    swing_foot_pos[:,0] = np.linspace(LIPM_model.left_foot_pos[0], left_foot_target_pos[0], swing_data_len)
    swing_foot_pos[:,1] = np.linspace(LIPM_model.left_foot_pos[1], left_foot_target_pos[1], swing_data_len)
    swing_foot_pos[1:swing_data_len-1, 2] = 0.1

# Initialize parameters
total_time = 10 # seconds
step_num = 0

theta = 0

step_to_cmdv = [10, 20, 30]
# COM_dvel_list = np.array([[1.0, 0.0],[0.0, 50.0],[2.0, 2.0],[-4.0, 0.]])
COM_dvel_list = np.array([[1.0, 0.0],[1.0, 0.0],[1.0, 0.0],[1.0, 0.0]])
                        
COM_dvel = COM_dvel_list[0]

for i in range(1, int(total_time/LIPM_model.dt)):

    # Update body (CoM) state: x_t, vx_t, y_t, vy_t
    LIPM_model.step()

    if LIPM_model.support_leg == 'left_leg':
        LIPM_model.right_foot_pos = [swing_foot_pos[j,0], swing_foot_pos[j,1], swing_foot_pos[j,2]]
    else:
        LIPM_model.left_foot_pos = [swing_foot_pos[j,0], swing_foot_pos[j,1], swing_foot_pos[j,2]]
    j += 1

    # record data
    COM_pos_x.append(LIPM_model.x_t + LIPM_model.support_foot_pos[0])
    COM_pos_y.append(LIPM_model.y_t + LIPM_model.support_foot_pos[1])
    eICP_x.append(LIPM_model.eICP_x)
    eICP_y.append(LIPM_model.eICP_y)
    COM_vel_x.append(LIPM_model.vx_t)
    COM_vel_y.append(LIPM_model.vy_t)
    COM_dvel_x.append(COM_dvel[0])
    COM_dvel_y.append(COM_dvel[1])
    left_foot_pos_x.append(LIPM_model.left_foot_pos[0])
    left_foot_pos_y.append(LIPM_model.left_foot_pos[1])
    left_foot_pos_z.append(LIPM_model.left_foot_pos[2])
    right_foot_pos_x.append(LIPM_model.right_foot_pos[0])
    right_foot_pos_y.append(LIPM_model.right_foot_pos[1])
    right_foot_pos_z.append(LIPM_model.right_foot_pos[2])
    
    if LIPM_model.support_leg == 'right_leg':
        left_step_command_x.append(LIPM_model.u_x)
        left_step_command_y.append(LIPM_model.u_y)
        left_step_command_z.append(0)
        right_step_command_x.append(np.nan)
        right_step_command_y.append(np.nan)
        right_step_command_z.append(np.nan)
    else:
        right_step_command_x.append(LIPM_model.u_x)
        right_step_command_y.append(LIPM_model.u_y)
        right_step_command_z.append(0)
        left_step_command_x.append(np.nan)
        left_step_command_y.append(np.nan)
        left_step_command_z.append(np.nan)

    rsupport_foot_pos_x = np.cos(theta)*support_foot_pos[0] + np.sin(theta)*support_foot_pos[1]
    rsupport_foot_pos_y = -np.sin(theta)*support_foot_pos[0] + np.cos(theta)*support_foot_pos[1]
    rprev_support_foot_pos_x = np.cos(theta)*prev_support_foot_pos[0] + np.sin(theta)*prev_support_foot_pos[1]
    rprev_support_foot_pos_y = -np.sin(theta)*prev_support_foot_pos[0] + np.cos(theta)*prev_support_foot_pos[1]
    
    step_length.append(rsupport_foot_pos_x - rprev_support_foot_pos_x)
    dstep_length.append(LIPM_model.s_d)
    step_width.append(np.abs(rsupport_foot_pos_y - rprev_support_foot_pos_y))
    dstep_width.append(LIPM_model.w_d)

    # switch the support leg
    if (i % swing_data_len == 0):
        j = 0

        prev_support_foot_pos = support_foot_pos
        # Switch the support leg / Update current body state (self.x_0, self.y_0, self.vx_0, self.vy_0)
        LIPM_model.switchSupportLeg() 
        step_num += 1

        support_foot_pos = np.array(LIPM_model.support_foot_pos)

        if step_num >= step_to_cmdv[2]:
            theta = np.arctan2(COM_dvel_list[3,1],COM_dvel_list[3,0])
            LIPM_model.s_d = np.linalg.norm(COM_dvel_list[3]) * LIPM_model.T
            COM_dvel = COM_dvel_list[3]
        elif step_num >= step_to_cmdv[1]:
            theta = np.arctan2(COM_dvel_list[2,1],COM_dvel_list[2,0])
            LIPM_model.s_d = np.linalg.norm(COM_dvel_list[2]) * LIPM_model.T
            COM_dvel = COM_dvel_list[2]
        elif step_num >= step_to_cmdv[0]:
            theta = np.arctan2(COM_dvel_list[1,1],COM_dvel_list[1,0])
            LIPM_model.s_d = np.linalg.norm(COM_dvel_list[1]) * LIPM_model.T
            COM_dvel = COM_dvel_list[1]
        else:
            theta = np.arctan2(COM_dvel_list[0,1],COM_dvel_list[0,0])
            LIPM_model.s_d = np.linalg.norm(COM_dvel_list[0]) * LIPM_model.T
            COM_dvel = COM_dvel_list[0]

        # Calculate the next step locations
        LIPM_model.calculateFootLocationForNextStepXcoMWorld(theta)
        # LIPM_model.calculateFootLocationForNextStepXcoMBase(theta)

        # calculate the foot positions for swing phase
        if LIPM_model.support_leg == 'left_leg':
            right_foot_target_pos = [LIPM_model.u_x, LIPM_model.u_y, 0]
            swing_foot_pos[:,0] = np.linspace(LIPM_model.right_foot_pos[0], right_foot_target_pos[0], swing_data_len)
            swing_foot_pos[:,1] = np.linspace(LIPM_model.right_foot_pos[1], right_foot_target_pos[1], swing_data_len)
            swing_foot_pos[1:swing_data_len-1, 2] = 0.1
        else:
            left_foot_target_pos = [LIPM_model.u_x, LIPM_model.u_y, 0]
            swing_foot_pos[:,0] = np.linspace(LIPM_model.left_foot_pos[0], left_foot_target_pos[0], swing_data_len)
            swing_foot_pos[:,1] = np.linspace(LIPM_model.left_foot_pos[1], left_foot_target_pos[1], swing_data_len)
            swing_foot_pos[1:swing_data_len-1, 2] = 0.1


# ------------------------------------------------- helper functions
def draw_quadrant(center, radius, start_u, end_u, start_v, end_v):
    # u, v are the parameterization variables
    u = np.linspace(start_u, end_u, 100)
    v = np.linspace(start_v, end_v, 100)
    u, v = np.meshgrid(u, v)
    x = center[0] + radius * np.sin(u) * np.cos(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(u)
    return x, y, z

# * ------------------------------------------------- Play the animation
data_len = len(COM_pos_x)
# print('--------- plot')
# fig = plt.figure(figsize=(10, 10))
# spec = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[2.5, 1])
# ax = fig.add_subplot(spec[0,0], projection ='3d')
# # ax.set_aspect('equal') # bugs
# ax.set_xlim(-1.0, 4.0)
# ax.set_ylim(-2.0, 2.0)
# ax.set_zlim(-0.01, 1.)
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_zlabel('z (m)')

# # view angles
# ax.view_init(20, -150)
# LIPM_3D_ani = LIPM_3D_Animate()
# ani_3D = FuncAnimation(fig, ani_3D_update, frames=range(1, data_len), init_func=ani_3D_init, interval=1.0/LIPM_model.dt, blit=False, repeat=True)

# # Add 2D plot for COM trajectory
# bx = fig.add_subplot(spec[1,0], autoscale_on=False)
# bx.set_xlim(-0.5, 5.0)
# bx.set_ylim(-0.8, 0.8)
# bx.set_aspect('equal')
# bx.set_xlabel('x (m)')
# bx.set_ylabel('y (m)')
# bx.grid(ls='--')

# COM_pos_str = 'COM = (%.2f, %.2f)'
# ani_text_COM_pos = bx.text(0.05, 0.9, '', transform=bx.transAxes)


# original_ani, = bx.plot(0, 0, marker='o', markersize=2, color='k')
# # left_foot_pos_ani, = bx.plot([], [], 'o', lw=2, color='b')
# COM_traj_ani, = bx.plot(COM_pos_x[0], COM_pos_y[0], markersize=2, color='g', linestyle='--')
# eICP_ani, = bx.plot(eICP_x[0], eICP_y[0], markersize=2, color='y', linestyle='--')
# COM_pos_ani, = bx.plot(COM_pos_x[0], COM_pos_y[0], marker='o', markersize=6, color='k')
# left_foot_pos_ani, = bx.plot([], [], 'o', markersize=10, color='c')
# right_foot_pos_ani, = bx.plot([], [], 'o', markersize=10, color='m')
# left_step_command_ani, = bx.plot([], [], 'x', markersize=10, color='b', linestyle='-')
# right_step_command_ani, = bx.plot([], [], 'x', markersize=10, color='r', linestyle='-')

# ani_2D = FuncAnimation(fig=fig, init_func=ani_2D_init, func=ani_2D_update, frames=range(1, data_len), interval=1.0/LIPM_model.dt, blit=False, repeat=True)

# # Add CoM velocity plot
# cx = fig.add_subplot(spec[0,1])
# cx.set_xlim(0, total_time/LIPM_model.dt)
# cx.set_ylim(min(min(COM_vel_x), min(COM_vel_y))-0.1, max(max(COM_vel_x), max(COM_vel_y))+0.1)
# cx.set_xlabel('time (s)')
# cx.set_ylabel('CoM velocity (m/s)')
# cx.set_xticklabels(np.linspace(0, total_time, 6))
# cx.grid(ls='--')

# COM_vel_x_ani, = cx.plot([], [], color='k', label='CoM velocity x')
# COM_dvel_x_ani, = cx.plot([], [], color='k', linestyle='--', label='desired CoM velocity x')
# COM_vel_y_ani, = cx.plot([], [], color='purple', label='CoM velocity y')
# COM_dvel_y_ani, = cx.plot([], [], color='purple', linestyle='--', label='desired CoM velocity y')
# cx.legend(loc='upper right')

# COM_vel_2D = FuncAnimation(fig=fig, init_func=COM_vel_2D_init, func=COM_vel_2D_update, frames=range(1, data_len), interval=1.0/LIPM_model.dt, blit=False, repeat=True)

# # Add analysis plot
# dx = fig.add_subplot(spec[1,1])
# dx.set_xlim(0, total_time/LIPM_model.dt)
# dx.set_ylim(min(min(step_length), min(step_width))-0.1, max(max(step_length), max(step_width))+0.1)
# dx.set_xlabel('time (s)')
# dx.set_ylabel('scale')
# dx.set_xticklabels(np.linspace(0, total_time, 6))
# dx.grid(ls='--')

# step_length_ani, = dx.plot([], [], color='gray', label='step length')
# dstep_length_ani, = dx.plot([], [], color='gray', linestyle='--', label='desired step length')
# step_width_ani, = dx.plot([], [], color='cyan', label='step width')
# dstep_width_ani, = dx.plot([], [], color='cyan', linestyle='--', label='desired step width')
# dx.legend(loc='upper right')

# step_params_2D = FuncAnimation(fig=fig, init_func=step_params_2D_init, func=step_params_2D_update, frames=range(1, data_len), interval=1.0/LIPM_model.dt, blit=False, repeat=True)

# plt.show()
# print('---------  Program terminated')


# * ------------------------------------------------- Save 3D animation (It takes a while to save the 3D animation)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# spec = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[2.5, 1])
# # bx = fig.add_subplot(111, autoscale_on=False)
# # ax.set_aspect('equal') # bugs
# ax.set_xlim(-1.0, 4.0)
# ax.set_ylim(-2.0, 2.0)
# ax.set_zlim(-0.01, 1.)
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_zlabel('z (m)')
# ax.set_box_aspect([1.4,0.6,0.66])
# # view angles
# ax.view_init(20, -130)
# # Get rid of the panes
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# # view angles
# LIPM_3D_ani = LIPM_3D_Animate()
# ani_3D = FuncAnimation(fig, ani_3D_update, frames=range(1, data_len), init_func=ani_3D_init, interval=1.0/LIPM_model.dt, blit=False, repeat=True)

# print("--------- Save the 3D animation")
# filepath = os.path.join(os.getcwd(), "LIPM_vt_3D.mp4")
# # COM_vel_2D.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])
# # step_params_2D.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])
# ani_3D.save(filepath, fps=1.0/LIPM_model.dt, extra_args=['-vcodec', 'libx264'])


# * ------------------------------------------------- Save 2D animation
# Add 2D plot for COM trajectory
fig = plt.figure(figsize=(10, 10))
bx = fig.add_subplot(111, autoscale_on=False)
bx.set_xlim(-0.5, 5.0)
bx.set_ylim(-0.8, 0.8)
bx.set_aspect('equal')
bx.set_xlabel('x (m)')
bx.set_ylabel('y (m)')
bx.grid(ls='--')

ani_text_COM_pos = bx.text(0.05, 0.9, '', transform=bx.transAxes)

# Define the patches for the circle and wedges outside of the init function so they can be updated later
circle = patches.Circle((COM_pos_x[0], COM_pos_y[0]), radius=0.05, edgecolor='black', fill=False)
wedge_top_right = patches.Wedge((COM_pos_x[0], COM_pos_y[0]), 0.05, 0, 90, color="black")
wedge_bottom_left = patches.Wedge((COM_pos_x[0], COM_pos_y[0]), 0.05, 180, 270, color="black")

original_ani, = bx.plot(0, 0, marker='o', markersize=2, color='k')
# left_foot_pos_ani, = bx.plot([], [], 'o', lw=2, color='b')
COM_traj_ani, = bx.plot(COM_pos_x[0], COM_pos_y[0], markersize=2, color='g', linestyle='--')
eICP_ani, = bx.plot(eICP_x[0], eICP_y[0], markersize=2, color='y', linestyle='--')
COM_pos_ani, = bx.plot(COM_pos_x[0], COM_pos_y[0], marker='o', markersize=6, color='k')
COM_circle_ani = bx.add_patch(circle)
COM_wedge_top_right_ani = bx.add_patch(wedge_top_right)
COM_wedge_bottom_left_ani = bx.add_patch(wedge_bottom_left)
vel_cmd_ani = bx.quiver(COM_pos_x[0], COM_pos_y[0], 0.15, 0, scale=1, color='#7FFF00')

left_foot_pos_ani, = bx.plot([], [], 'o', markersize=10, color='c')
right_foot_pos_ani, = bx.plot([], [], 'o', markersize=10, color='m')
left_step_command_ani, = bx.plot([], [], 'x', markersize=10, color='b', linestyle='-')
right_step_command_ani, = bx.plot([], [], 'x', markersize=10, color='r', linestyle='-')

ani_2D = FuncAnimation(fig=fig, init_func=ani_2D_init, func=ani_2D_update, frames=range(1, data_len), interval=1.0/LIPM_model.dt, blit=False, repeat=True)

print("--------- Save the 2D animation")
filepath = os.path.join(os.getcwd(), "LIPM_vt_2D.mp4")
# COM_vel_2D.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])
# step_params_2D.save(filepath, fps=self.fps, extra_args=['-vcodec', 'libx264'])
ani_2D.save(filepath, fps=1.0/LIPM_model.dt, extra_args=['-vcodec', 'libx264'])


# * ------------------------------------------------- Save static plot
start, end = 0, 499
# start, end = 130, 499
choice = 185

# * 3D plot
fig3d = plt.figure(figsize=(10, 10))
ax3d = fig3d.add_subplot(111, projection='3d')
spec = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[2.5, 1])
ax3d.set_xlim(3.6, 5.0)
ax3d.set_ylim(-1.0, -0.4)
ax3d.set_zlim(-0.01, 0.65)
ax3d.set_xlabel('x (m)')
ax3d.set_ylabel('y (m)')
ax3d.set_zlabel('z (m)')
# ax3d.set_aspect('auto')
ax3d.set_box_aspect([1.4,0.6,0.66])
# view angles
ax3d.view_init(20, -130)
# Get rid of the panes
ax3d.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3d.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3d.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

COM_pos = np.array([COM_pos_x[choice], COM_pos_y[choice], LIPM_model.zc])

# Radius of the sphere
radius = 0.05

# Quadrant 1: top right front
x, y, z = draw_quadrant(COM_pos, radius, 0, np.pi/2, 0, np.pi/2)
ax3d.plot_surface(x, y, z, color='black')

# Quadrant 2: top left front
x, y, z = draw_quadrant(COM_pos, radius, 0, np.pi/2, np.pi/2, np.pi)
ax3d.plot_surface(x, y, z, color='white')

# Quadrant 3: top left back
x, y, z = draw_quadrant(COM_pos, radius, 0, np.pi/2, np.pi, 3*np.pi/2)
ax3d.plot_surface(x, y, z, color='black')

# Quadrant 4: top right back
x, y, z = draw_quadrant(COM_pos, radius, 0, np.pi/2, 3*np.pi/2, 2*np.pi)
ax3d.plot_surface(x, y, z, color='white')

# Quadrant 5: bottom right front
x, y, z = draw_quadrant(COM_pos, radius, np.pi/2, np.pi, 0, np.pi/2)
ax3d.plot_surface(x, y, z, color='white')

# Quadrant 6: bottom left front
x, y, z = draw_quadrant(COM_pos, radius, np.pi/2, np.pi, np.pi/2, np.pi)
ax3d.plot_surface(x, y, z, color='black')

# Quadrant 7: bottom left back
x, y, z = draw_quadrant(COM_pos, radius, np.pi/2, np.pi, np.pi, 3*np.pi/2)
ax3d.plot_surface(x, y, z, color='white')

# Quadrant 8: bottom right back
x, y, z = draw_quadrant(COM_pos, radius, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi)
ax3d.plot_surface(x, y, z, color='black')

ax3d.quiver(COM_pos_x[choice], COM_pos_y[choice], LIPM_model.zc + 0.1, 0.2, 0, 0, length=1, color='#7FFF00', label="Velocity Command")

left_leg = np.zeros((3,2))
left_leg[:,0] = COM_pos_x[choice], COM_pos_y[choice], LIPM_model.zc
left_leg[:,1] = left_foot_pos_x[choice], left_foot_pos_y[choice], left_foot_pos_z[choice]
ax3d.plot(left_leg[0,:], left_leg[1,:], left_leg[2,:], linewidth=2, color='k', linestyle='-')

right_leg = np.zeros((3,2))
right_leg[:,0] = COM_pos_x[choice], COM_pos_y[choice], LIPM_model.zc
right_leg[:,1] = right_foot_pos_x[choice], right_foot_pos_y[choice], right_foot_pos_z[choice]
ax3d.plot(right_leg[0,:], right_leg[1,:], right_leg[2,:], linewidth=2, color='k', linestyle='-')

ax3d.plot(left_foot_pos_x[choice], left_foot_pos_y[choice], left_foot_pos_z[choice], marker='o', markersize=10, color='c')
ax3d.plot(right_foot_pos_x[choice], right_foot_pos_y[choice], right_foot_pos_z[choice], marker='o', markersize=10, color='m')


def truncate_data(data, limit):
    data_copy = np.copy(data)
    # truncate_data = np.maximum(np.minimum(data, limit[1]), limit[0])
    # truncate_data = np.maximum(data, limit[0])
    data_copy[data_copy < limit[0]] = np.nan
    data_copy[data_copy > limit[1]] = np.nan
    return data_copy

ax3d.plot(COM_pos_x[start:end], COM_pos_y[start:end], LIPM_model.zc, color='green', label='CoM Trajectory')
ax3d.plot(COM_pos_x[start:end], COM_pos_y[start:end], 0, color='green', linestyle='--', alpha=0.8, label='Projected CoM Trajectory')
ax3d.plot(eICP_x[start:end], eICP_y[start:end], 0, color='y', linestyle='--', alpha=0.8, label='ICP Trajectory')
ax3d.scatter(left_step_command_x[start:end], left_step_command_y[start:end], left_step_command_z[start:end], color='b', marker='x', s=60, linewidths=1.5, label='Desired Left Step')
ax3d.scatter(right_step_command_x[start:end], right_step_command_y[start:end], right_step_command_z[start:end], color='r', marker='x', s=60, linewidths=1.5, label='Desired Right Step')

# ax3d.plot(truncate_data(COM_pos_x[start:end], ax3d.get_xlim()), truncate_data(COM_pos_y[start:end], ax3d.get_ylim()), LIPM_model.zc, color='green', label='CoM Trajectory')
# ax3d.plot(truncate_data(COM_pos_x[start:end], ax3d.get_xlim()), truncate_data(COM_pos_y[start:end], ax3d.get_ylim()), 0, color='green', linestyle='--', alpha=0.8, label='Projected CoM Trajectory')
# ax3d.plot(truncate_data(eICP_x[start:end], ax3d.get_xlim()), truncate_data(eICP_y[start:end], ax3d.get_ylim()), 0, color='y', linestyle='--', alpha=0.8, label='ICP Trajectory')
# ax3d.scatter(truncate_data(left_step_command_x[start:end], ax3d.get_xlim()), truncate_data(left_step_command_y[start:end], ax3d.get_ylim()), left_step_command_z[start:end], color='b', marker='x', s=60, linewidths=1.5, label='Left Step Command')
# ax3d.scatter(truncate_data(right_step_command_x[start:end], ax3d.get_xlim()), truncate_data(right_step_command_y[start:end], ax3d.get_ylim()), right_step_command_z[start:end], color='r', marker='x', s=60, linewidths=1.5, label='Right Step Command')

# ax3d.legend(loc='upper right')
ax3d.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.95))
ax3d.set_clip_on(True)

fig3d.savefig(os.path.join(os.getcwd(), "LIP_3D.pdf"), format='pdf', bbox_inches='tight')

# * 2D plot
fig = plt.figure(figsize=(10, 10))
ax2d = fig.add_subplot(111, autoscale_on=False)
ax2d.set_xlim(3.6, 5.0)
ax2d.set_ylim(-0.95, -0.45)
ax2d.set_xlabel('x (m)')
ax2d.set_ylabel('y (m)')
ax2d.grid(ls='--')
ax2d.set_aspect('equal')

circle = patches.Circle((COM_pos_x[choice], COM_pos_y[choice]), radius=0.05, edgecolor='black', fill=False)
# Add the circle to the Axes
ax2d.add_patch(circle)
# Fill the top right quadrant
ax2d.add_patch(patches.Wedge((COM_pos_x[choice], COM_pos_y[choice]), 0.05, 0, 90, color="black"))
# Fill the bottom left quadrant
ax2d.add_patch(patches.Wedge((COM_pos_x[choice], COM_pos_y[choice]), 0.05, 180, 270, color="black"))

ax2d.quiver(COM_pos_x[choice], COM_pos_y[choice], 0.15, 0, scale=1, color='#7FFF00')

ax2d.plot(COM_pos_x[start:end], COM_pos_y[start:end], markersize=2, color='g', linestyle='--')
ax2d.plot(eICP_x[start:end], eICP_y[start:end], markersize=2, color='y', linestyle='--')
ax2d.plot(left_step_command_x[start:end], left_step_command_y[start:end], 'x', markersize=8, color='b', markeredgewidth=2)
ax2d.plot(right_step_command_x[start:end], right_step_command_y[start:end], 'x', markersize=8, color='r', markeredgewidth=2)

# Add annotation
wd_arrow = FancyArrowPatch((3.8, -0.5), (3.8, -0.87),
                        arrowstyle='<->',  # Two-headed arrow
                        mutation_scale=20, # Size of arrow head
                        color='black',     # Color of the arrow
                        lw=1) 
ax2d.text(3.76, -0.63, r'$w_d$', ha='center', va='center')
    
sd_arrow = FancyArrowPatch((3.8, -0.5), (4.1, -0.5),
                        arrowstyle='<->',  # Two-headed arrow
                        mutation_scale=20, # Size of arrow head
                        color='black',     # Color of the arrow
                        lw=1)
ax2d.text(3.95, -0.53, r'$s_d$', ha='center', va='center')

bx_arrow = FancyArrowPatch((3.8, -0.87), (3.93, -0.87),
                        arrowstyle='<->',  # Two-headed arrow
                        mutation_scale=20, # Size of arrow head
                        color='black',     # Color of the arrow
                        lw=1)
ax2d.text(3.87, -0.9, r'$b_x$', ha='center', va='center')

by_arrow = FancyArrowPatch((3.93, -0.87), (3.93, -0.8),
                        arrowstyle='<->',  # Two-headed arrow
                        mutation_scale=20, # Size of arrow head
                        color='black',     # Color of the arrow
                        lw=1)
ax2d.text(3.97, -0.85, r'$b_y$', ha='center', va='center')

ax2d.text(3.66, -0.88, r'($p_x$, $p_y$)')
ax2d.text(3.95, -0.8, r'($\xi_x$, $\xi_y$)')

ax2d.add_patch(wd_arrow)
ax2d.add_patch(sd_arrow)
ax2d.add_patch(bx_arrow)
ax2d.add_patch(by_arrow)

fig.savefig(os.path.join(os.getcwd(), "LIP_2D.pdf"), format='pdf', bbox_inches='tight')

plt.show()
print('---------  Program terminated')
