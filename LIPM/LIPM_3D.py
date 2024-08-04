# -*-coding:UTF-8 -*-
import numpy as np

# Deinition of 3D Linear Inverted Pendulum
class LIPM3D:
    def __init__(self,
                 dt=0.001,
                 T=1.0,
                 T_d = 0.4,
                 s_d = 0.5,
                 w_d = 0.4,
                 support_leg='left_leg'):
        self.dt = dt
        self.t = 0
        self.T = T # step time
        self.T_d = T # desired step duration
        self.s_d = s_d # desired step length
        self.w_d = w_d # desired step width

        self.eICP_x = 0 # ICP location x
        self.eICP_y = 0 # ICP location y
        self.u_x = 0 # step location x
        self.u_y = 0 # step location y

        # COM initial state
        self.x_0 = 0
        self.vx_0 = 0
        self.y_0 = 0
        self.vy_0 = 0

        # COM real-time state
        self.x_t = 0
        self.vx_t = 0
        self.y_t = 0
        self.vy_t = 0

        self.support_leg = support_leg
        self.support_foot_pos = [0.0, 0.0, 0.0]
        self.left_foot_pos = [0.0, 0.0, 0.0]
        self.right_foot_pos = [0.0, 0.0, 0.0]
        self.COM_pos = [0.0, 0.0, 0.0]
    
    def initializeModel(self, COM_pos, left_foot_pos, right_foot_pos):
        self.COM_pos = COM_pos

        if self.support_leg == 'left_leg':
            self.left_foot_pos = left_foot_pos
            self.right_foot_pos = left_foot_pos
            self.support_foot_pos = left_foot_pos
        elif self.support_leg == 'right_leg':
            self.left_foot_pos = right_foot_pos
            self.right_foot_pos = right_foot_pos
            self.support_foot_pos = right_foot_pos

        self.zc = self.COM_pos[2]
        self.w_0 = np.sqrt(9.81 / self.zc)
    
    def step(self):
        self.t += self.dt
        t = self.t

        self.x_t = self.x_0*np.cosh(t*self.w_0) + self.vx_0*np.sinh(t*self.w_0)/self.w_0
        self.vx_t = self.x_0*self.w_0*np.sinh(t*self.w_0) + self.vx_0*np.cosh(t*self.w_0)

        self.y_t = self.y_0*np.cosh(t*self.w_0) + self.vy_0*np.sinh(t*self.w_0)/self.w_0
        self.vy_t = self.y_0*self.w_0*np.sinh(t*self.w_0) + self.vy_0*np.cosh(t*self.w_0)

    def calculateXfVf(self):
        x_f = self.x_0*np.cosh(self.T*self.w_0) + self.vx_0*np.sinh(self.T*self.w_0)/self.w_0
        vx_f = self.x_0*self.w_0*np.sinh(self.T*self.w_0) + self.vx_0*np.cosh(self.T*self.w_0)

        y_f = self.y_0*np.cosh(self.T*self.w_0) + self.vy_0*np.sinh(self.T*self.w_0)/self.w_0
        vy_f = self.y_0*self.w_0*np.sinh(self.T*self.w_0) + self.vy_0*np.cosh(self.T*self.w_0)

        return x_f, vx_f, y_f, vy_f

    def calculateFootLocationForNextStepXcoMWorld(self, theta=0.):
        x_f, vx_f, y_f, vy_f = self.calculateXfVf()
        x_f_world = x_f + self.support_foot_pos[0]
        y_f_world = y_f + self.support_foot_pos[1]
        self.eICP_x = x_f_world + vx_f/self.w_0
        self.eICP_y = y_f_world + vy_f/self.w_0
        b_x = self.s_d / (np.exp(self.w_0*self.T_d) - 1)
        b_y = self.w_d / (np.exp(self.w_0*self.T_d) + 1)

        original_offset_x = -b_x
        original_offset_y = -b_y if self.support_leg == "left_leg" else b_y 
        offset_x = np.cos(theta) * original_offset_x - np.sin(theta) * original_offset_y
        offset_y = np.sin(theta) * original_offset_x + np.cos(theta) * original_offset_y

        self.u_x = self.eICP_x + offset_x
        self.u_y = self.eICP_y + offset_y

    def calculateFootLocationForNextStepXcoMBase(self, theta=0.):
        x_f, vx_f, y_f, vy_f = self.calculateXfVf()
        x_f_world = x_f + self.support_foot_pos[0]
        y_f_world = y_f + self.support_foot_pos[1]
        self.eICP_x = x_f_world + vx_f/self.w_0
        self.eICP_y = y_f_world + vy_f/self.w_0
        b_x = self.s_d / (np.exp(self.w_0*self.T_d) - 1)
        b_y = self.w_d / (np.exp(self.w_0*self.T_d) + 1)

        original_offset_x = -b_x
        original_offset_y = -b_y if self.support_leg == "left_leg" else b_y 
        offset_x = np.cos(theta) * original_offset_x - np.sin(theta) * original_offset_y
        offset_y = np.sin(theta) * original_offset_x - np.cos(theta) * original_offset_y

        self.u_x = self.eICP_x + offset_x
        self.u_y = self.eICP_y + offset_y

    def switchSupportLeg(self):
        if self.support_leg == 'left_leg':
            print('\n---- switch the support leg to the right leg')
            self.support_leg = 'right_leg'
            COM_pos_x = self.x_t + self.left_foot_pos[0]
            COM_pos_y = self.y_t + self.left_foot_pos[1]
            self.x_0 = COM_pos_x - self.right_foot_pos[0]
            self.y_0 = COM_pos_y - self.right_foot_pos[1]
            self.support_foot_pos = self.right_foot_pos
        elif self.support_leg == 'right_leg':
            print('\n---- switch the support leg to the left leg')
            self.support_leg = 'left_leg'
            COM_pos_x = self.x_t + self.right_foot_pos[0]
            COM_pos_y = self.y_t + self.right_foot_pos[1]
            self.x_0 = COM_pos_x - self.left_foot_pos[0]
            self.y_0 = COM_pos_y - self.left_foot_pos[1]
            self.support_foot_pos = self.left_foot_pos

        self.t = 0
        self.vx_0 = self.vx_t
        self.vy_0 = self.vy_t
