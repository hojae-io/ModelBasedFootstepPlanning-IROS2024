from setuptools import find_packages
from distutils.core import setup

setup(
    name='gpuGym',
    version='1.0.1',
    author='Biomimetic Robotics Lab',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Isaac Gym environments for Legged Robots',
    install_requires=['isaacgym',
                      'matplotlib',
                      'pandas',
                      'tensorboard',
                      'setuptools==59.5.0',
		              'torch>=1.4.0',
		              'torchvision>=0.5.0',
		              'numpy>=1.16.4']
)
