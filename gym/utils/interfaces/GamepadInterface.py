from isaacgym import gymapi
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class GamepadInterface():
    def __init__(self, env):
        pygame.init()
        pygame.joystick.init()
        print("__________________________________________________________")
        if pygame.joystick.get_count() == 1:
            print("Using gamepad interface, overriding default comand settings")
            print("left joystick: forward, strafe left, "
                  "backward, strafe right")
            print("right joystick (left/right): yaw left/right")
            print("back button: reset environments")
            print("start button: quit")
            self.joystick = pygame.joystick.Joystick(0)

            env.commands[:] = 0.
            # never resample
            env.cfg.commands.resampling_time = env.max_episode_length + 1
            self.max_vel_backward = env.cfg.commands.ranges.lin_vel_x[0]
            self.max_vel_forward = env.cfg.commands.ranges.lin_vel_x[1]
            self.max_vel_sideways = env.cfg.commands.ranges.lin_vel_y
            self.max_yaw_vel = env.cfg.commands.ranges.yaw_vel
            # set to 0
            env.command_ranges["lin_vel_x"] = [0., 0.]
            env.command_ranges["lin_vel_y"] = 0.
            env.command_ranges["yaw_vel"] = 0.
        elif pygame.joystick.get_count() > 1:
            print("WARNING: you have more than one gamepad plugged in."
                  "Please unplug one.")
        else:
            print("WARNING: failed to initialize gamepad.")
        print("__________________________________________________________")


    def update(self, env):
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 7:  # press start
                    exit()
                if event.button == 6:  # press back
                    env.episode_length_buf[:] = env.max_episode_length+1
            if event.type == pygame.JOYAXISMOTION:
                # left joystick
                if event.axis == 1:  # up-down
                    if event.value >= 0:
                        env.commands[:, 0] = self.max_vel_backward*event.value
                    else:
                        env.commands[:, 0] = -self.max_vel_forward*event.value
                if event.axis == 0:  # left_right
                    env.commands[:, 1] = -self.max_vel_sideways*event.value
                # right joystick
                if event.axis == 3:  # left-right
                    env.commands[:, 2] = -self.max_yaw_vel*event.value
