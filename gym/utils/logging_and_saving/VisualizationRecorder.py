from subprocess import Popen, PIPE
import subprocess
import os
import mss
from gym import LEGGED_GYM_ROOT_DIR
from isaacgym import gymapi


class VisualizationRecorder():
    env = None
    frames = []
    framerate = 50
    frame_sampling_rate = 0
    target_window = None
    experiment_name = None
    run_name = None

    def __init__(self, env, experiment_name, log_dir, framerate=50):
        self.env = env
        self.framerate = framerate
        self.experiment_name = experiment_name
        self.run_name = os.path.basename(log_dir)
        self.target_window = self.getWindowGeometry()
        self.framerate = max(50., self.framerate)
        ctrl_frequency = int(1 / round(self.env.dt, 3))
        self.frame_sampling_rate = max(1, int(ctrl_frequency / self.framerate))
        self.sampling_frequency = ctrl_frequency / self.frame_sampling_rate
            
        self.playback_speed = self.sampling_frequency / self.framerate

    def update(self, sim_iter):
        self.captureFrame(sim_iter)
        if self.env.record_done:
            self.save()
            exit()

    def getWindowGeometry(self):
        try:
            output = subprocess.check_output(['wmctrl', '-lG']).decode('utf-8')
            lines = output.splitlines()
            for line in lines:
                if line.split(None, 4)[-1].split(" ")[-2:] == ['Isaac', 'Gym']:
                    x, y, width, height = line.split()[2:6]
                    monitor = {'top': int(y), 'left': int(x),
                               'width': int(width), 'height': int(height)}
                    return monitor
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def captureFrame(self, sim_iter):
        with mss.mss() as sct:
            if sim_iter % self.frame_sampling_rate == 0:
                # Capture the screenshot
                try:
                    screenshot = sct.grab(self.target_window)
                    self.frames.append(screenshot)
                except:
                    print("Please install wm-ctrl (sudo apt-get install  \
                           wmctrl) if you want to record at real time.")
                    exit()

    def save(self):
        print("Converting recorded frames to video...")
        folderpath = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                                  self.experiment_name, 'videos')
        filepath = os.path.join(folderpath, self.run_name + ".mp4")
        os.makedirs(folderpath, exist_ok=True)

        # Use FFmpeg directly to pipe frames
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f"{self.frames[0].size.width}x{self.frames[0].size.height}",
            '-pix_fmt', 'rgb24',
            '-r', str(self.framerate),  # Frame rate
            '-i', '-',  # Read from pipe
            '-an',  # No audio
            '-c:v', 'libx264',
            '-vf', f"setpts=PTS/{self.playback_speed}",
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            '-preset', 'slow',  # Slow compression for better quality
            '-profile:v', 'high',  # High profile
            filepath
        ]
        with Popen(ffmpeg_cmd, stdin=PIPE) as ffmpeg_proc:
            for screenshot in self.frames:
                ffmpeg_proc.stdin.write(screenshot.rgb)
        os.system("xdg-open " + filepath)
