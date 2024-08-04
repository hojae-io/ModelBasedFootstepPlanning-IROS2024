from isaacgym import gymapi
import torch
class BaseKeyboardInterface:
    def __init__(self, env):
        self.env = env
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer, 
                                                     gymapi.KEY_V, "toggle_viewer_sync")
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_R, "RESET")
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_X, 'screenshot')
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_W, 'record')
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer, 
                                                     gymapi.KEY_ESCAPE, "QUIT")

        print("______________________________________________________________")
        print("Base keyboard interface")
        print("V: stop viewer sync")
        print("R: reset environments")
        print("X: take screenshot")
        print("W: record")
        print("ESC: quit")
        print("______________________________________________________________")

    def update(self):
        for evt in self.env.gym.query_viewer_action_events(self.env.viewer):
            # evt.value = 0 means key is released. Use only when key is pressed
            if evt.value == 0.: 
                continue
            if evt.action == "toggle_viewer_sync":
                self.env.enable_viewer_sync = not self.env.enable_viewer_sync
            elif evt.action == "RESET":
                self.env.reset()
            elif evt.action == 'screenshot':
                self.env.screenshot = True
            elif evt.action == 'record':
                self.env.record_done = True
            elif evt.action == "QUIT":
                exit()

class KeyboardInterface(BaseKeyboardInterface):
    def __init__(self, env):
        super().__init__(env)
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_L, 'forward')
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_H, 'backward')
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_K, 'left')
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_J, 'right')
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_I, 'yaw_left')
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_U, 'yaw_right')
        
        self.offset = 0.2
        
        print("______________________________________________________________")
        print("Keyboard interface")
        print(f"Commands are modified by {self.offset}")
        print("LH: forward, backward\n"
              "KJ: left, right")
        print("IU: yaw left/right")
        print("______________________________________________________________")

    def update(self):
        for evt in self.env.gym.query_viewer_action_events(self.env.viewer):
            # evt.value = 0 means key is released. Use only when key is pressed
            if evt.value == 0.: 
                continue
            if evt.action == 'forward':
                self.env.commands[:, 0] += self.offset
            elif evt.action == 'backward':
                self.env.commands[:, 0] -= self.offset
            elif evt.action == 'left':
                self.env.commands[:, 1] += self.offset
            elif evt.action == 'right':
                self.env.commands[:, 1] -= self.offset
            elif evt.action == 'yaw_right':
                self.env.commands[:, 2] += self.offset
            elif evt.action == 'yaw_left':
                self.env.commands[:, 2] -= self.offset
            elif evt.action == "toggle_viewer_sync":
                self.env.enable_viewer_sync = not self.env.enable_viewer_sync
            elif evt.action == "RESET":
                self.env.reset()
            elif evt.action == "QUIT":
                exit()
            self.print_commands()

    def print_commands(self):
        print("______________________________________________________________")
        print("  commands: ", self.env.commands)
        
class VanillaKeyboardInterface(KeyboardInterface):
    def __init__(self, env):
        super().__init__(env)
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_M, 'increase')
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_N, 'decrease')   
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_S, 'offset_inc')
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_A, 'offset_dec')     
        
        print("______________________________________________________________")
        print("Vanilla Keyboard interface")
        print("MN: step period increase/decrease")
        print("SA: offset increase/decrease")
        print("______________________________________________________________")

    def update(self):
        for evt in self.env.gym.query_viewer_action_events(self.env.viewer):
            # evt.value = 0 means key is released. Use only when key is pressed
            if evt.value == 0.: 
                continue
            if evt.action == 'forward':
                self.env.commands[:, 0] += self.offset
            elif evt.action == 'backward':
                self.env.commands[:, 0] -= self.offset
            elif evt.action == 'left':
                self.env.commands[:, 1] += self.offset
            elif evt.action == 'right':
                self.env.commands[:, 1] -= self.offset
            elif evt.action == 'yaw_right':
                self.env.commands[:, 2] += self.offset
            elif evt.action == 'yaw_left':
                self.env.commands[:, 2] -= self.offset
            elif evt.action == 'increase':
                self.env.step_period += 1
            elif evt.action == 'decrease':
                self.env.step_period -= 1
            elif evt.action == 'offset_inc':
                self.offset += 0.1
            elif evt.action == 'offset_dec':
                self.offset -= 0.1
            elif evt.action == "toggle_viewer_sync":
                self.env.enable_viewer_sync = not self.env.enable_viewer_sync
            elif evt.action == "RESET":
                self.env.reset()
            elif evt.action == 'screenshot':
                self.env.screenshot = True
            elif evt.action == 'record':
                self.env.record_done = True
            elif evt.action == "QUIT":
                exit()
            self.print_commands()

    def print_commands(self):
        print("______________________________________________________________")
        print("   commands: ", self.env.commands)
        print("step_period: ", self.env.step_period)
        print("     offset: ", self.offset)

class ControllerKeyboardInterface(KeyboardInterface):
    def __init__(self, env):
        super().__init__(env)
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_M, 'increase')
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer,
                                                     gymapi.KEY_N, 'decrease')
        print("______________________________________________________________")
        print("Controller Keyboard interface")
        print("MN: step period increase/decrease")
        print("______________________________________________________________")

    def update(self):
        for evt in self.env.gym.query_viewer_action_events(self.env.viewer):
            # evt.value = 0 means key is released. Use only when key is pressed
            if evt.value == 0: 
                continue
            if evt.action == 'forward':
                self.env.command_angle += torch.pi / 4 # 0.2
            elif evt.action == 'backward':
                self.env.command_angle -= torch.pi / 4 # 0.2 
            elif evt.action == 'left':
                self.env.command_radius += 0.1 
            elif evt.action == 'right':
                self.env.command_radius -= 0.1 
            elif evt.action == 'yaw_right':
                self.env.commands[:, 2] += 0.2
            elif evt.action == 'yaw_left':
                self.env.commands[:, 2] -= 0.2 
            elif evt.action == 'increase':
                self.env.step_period += 1
            elif evt.action == 'decrease':
                self.env.step_period -= 1
            elif evt.action == "toggle_viewer_sync":
                self.env.enable_viewer_sync = not self.env.enable_viewer_sync
            elif evt.action == "RESET":
                self.env.reset()
                self.env.manual_reset_flag = True
            elif evt.action == "QUIT":
                exit()
            self.print_commands()

    def print_commands(self):
        print("______________________________________________________________")
        print("   command_angle: ", self.env.command_angle)
        print("  command_radius: ", self.env.command_radius)
        print("  commands[:, 2]: ", self.env.commands[:, 2])
        print("     step_period: ", self.env.step_period)

        # print("______________________________________________________________")
        # print("   commands:", self.env.commands)
        # print("step_period:", self.env.step_period)

        
class XCoMKeyboardInterface(VanillaKeyboardInterface):
    def __init__(self, env):
        super().__init__(env)

        print("______________________________________________________________")
        print("XCoM Keyboard interface")
        print("MN: Desired step width increase/decrease")
        print("______________________________________________________________")

    def update(self):
        for evt in self.env.gym.query_viewer_action_events(self.env.viewer):
            # evt.value = 0 means key is released. Use only when key is pressed
            if evt.value == 0.: 
                continue
            if evt.action == 'forward':
                self.env.commands[:, 0] += self.offset
            elif evt.action == 'backward':
                self.env.commands[:, 0] -= self.offset
            elif evt.action == 'left':
                self.env.commands[:, 1] += self.offset
            elif evt.action == 'right':
                self.env.commands[:, 1] -= self.offset
            elif evt.action == 'yaw_right':
                self.env.commands[:, 2] += self.offset
            elif evt.action == 'yaw_left':
                self.env.commands[:, 2] -= self.offset
            elif evt.action == 'increase':
                self.env.dstep_width += 0.1
            elif evt.action == 'decrease':
                self.env.dstep_width -= 0.1
            elif evt.action == 'offset_inc':
                self.offset += 0.1
            elif evt.action == 'offset_dec':
                self.offset -= 0.1
            elif evt.action == "toggle_viewer_sync":
                self.env.enable_viewer_sync = not self.env.enable_viewer_sync
            elif evt.action == "RESET":
                self.env.reset()
            elif evt.action == 'screenshot':
                self.env.screenshot = True
            elif evt.action == 'record':
                self.env.record_done = True
            elif evt.action == "QUIT":
                exit()
            self.print_commands()

    def print_commands(self):
        print("______________________________________________________________")
        print("commands: ", self.env.commands)
        print("dstep width: ", self.env.dstep_width)
        print("offset: ", self.offset)
