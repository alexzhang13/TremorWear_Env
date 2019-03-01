import gym
import random
import math
import numpy as np

from environment import movements

import importlib

# Case-by-Case Simulations
_REST = "RestNoTremor"
_REST_T = "RestTremor"
_CONSTFREEHAND = "ConstantFreeHandNoTremor"
_CONSTFREEHAND_T = "ConstantFreeHandTremor"
_FREEHAND = "FreeHandNoTremor"
_FREEHAND_T = "FreeHandTremor"
_FREEROT = "FreeHandRotNoTremor"
_FREEROT_T = "FreeHandTremor"

# Constants
IMU_UPPER_BOUND = 16384
IMU_LOWER_BOUND = -16384
LOCKING_UPPER_BOUND = 1.0
LOCKING_LOWER_BOUND = 0.0
DEVICE_HZ = 60.0 # Operations per second
DELTA_T = 1/DEVICE_HZ # Change in time per episode

simulated_action_space = [
    _REST,
    _REST_T,
    _CONSTFREEHAND,
    _CONSTFREEHAND_T,
    _FREEHAND,
    _FREEHAND_T,
    _FREEROT,
    _FREEROT_T
]

class TremorEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def __init__(self):
        # Initialize Action + Obs space
        self.action_space = []
        self.observation_space = []

        # Initialize Locking Mechanism Information
        self.lock = 0.0  # initialize locking mechanism as "unlocked"

        # Initialize Spatiotemporal information from start to end of episode [ground truth]
        self.init_constants()
        self.steps = 0 # number of passed steps in the episode (1 step = 1 delta_t in time)
        self.maxSteps = 1000 # max number of steps per episode

        # Initialize Sensor Info (Env outputs Spatiotemporal Info which must be integrated to the Sensor Info)
        self.sensor_info = SensorData()

        # Initialize Random Seeds for Replayability
        np.random.seed(42)
        random.seed(42)

        # Reset environment + start simulation
        self._simulated_tremor = None
        self.reset()

    # Simulation takes a step forward in time
    def step(self, action):
        if self.steps >= self.maxSteps:
            done = True # end episode
        else:
            done = False

        # Update Obs Feedback from Integrated SpatioTemporalInfo
        self.sensor_info.integrate_update(self._simulated_tremor, self.st_info, self.steps+2)
        self.steps += 1

        # Take Action and Apply to Simulation // Update Obs Feedback Again


        # *For Debugging: Fill Info
        info = self.st_info

        return self.get_obs(), self.get_reward(), done, info

    # Update sensor info at next time step (generated simulation)
    def generate_stdata(self, movement, step):
        next_st_data = SpatioTemporalData()

        next_st_data.x = movement.x(self.axi, self.vxi, self.xi, step*DELTA_T)
        next_st_data.y = movement.y(self.ayi, self.vyi, self.yi, step*DELTA_T)
        next_st_data.z = movement.z(self.azi, self.vzi, self.zi, step*DELTA_T)
        next_st_data.roll = movement.roll(self.vx_angular, self.rolli, step*DELTA_T)
        next_st_data.pitch = movement.pitch(self.vy_angular, self.pitchi, step*DELTA_T)
        next_st_data.yaw = movement.yaw(self.vz_angular, self.yawi, step*DELTA_T)

        return next_st_data

    def get_obs(self):
        _obs = [self.sensor_info.vx, self.sensor_info.vy, self.sensor_info.vz, self.sensor_info.ax, self.sensor_info.ay
                ,self.sensor_info.az, self.sensor_info.vx_angular, self.sensor_info.vy_angular
                ,self.sensor_info.vz_angular, self.sensor_info.flex_angle, self.sensor_info.pressure_output]
        return _obs

    def reset(self):
        # reset and randomize simulated action/tremor
        self._simulated_tremor = self.init_movement()
        # reset environment variables
        self.lock = 0.0
        self.steps = 0  # number of passed steps in the episode (1 step = 1 delta_t in time)
        self.sensor_info.reset()

        # generate a sequence of events [ground truth] of size (maxsteps)+2
        self.st_info = [SpatioTemporalData()]
        for i in range(self.maxSteps+5):
            spatiotemporal = self.generate_stdata(self._simulated_tremor, i+1)
            self.st_info.append(spatiotemporal)

    def render(self, mode='human', close=False):
        pass

    # TODO: Create Reward Function
    # reward for step [based on how far off the tremor was suppressed]
    def get_reward(self):
        
        return 1

    # initialize random movement at the start of each episode
    def init_movement(self):
        # Initialize Simulated Env (Pick a case randomly)
        i = random.randint(0, 7) # random.randint produces a number between a,b INCLUSIVE (as opposed to np)
        environment = __import__("environment")
        gym_tremor = getattr(environment, "gym_tremor")
        env = getattr(gym_tremor, "envs")
        movements = getattr(env, "movements")
        _class = getattr(movements, simulated_action_space[i])
        _instance = _class(self.t_amp, self.t_freq)

        return _instance
    def init_constants(self):
        # spatial info
        self.axi = 0
        self.ayi = 0
        self.azi = 0
        self.vxi = 0
        self.vyi = 0
        self.vzi = 0
        self.xi = 0
        self.yi = 0
        self.zi = 0

        # rotational info
        self.rolli = 0
        self.pitchi = 0
        self.yawi = 0
        self.vx_angular = 0
        self.vy_angular = 0
        self.vz_angular = 0

        # tremor info
        self.t_amp = 2 # units of
        self.t_freq = 2*math.pi*4 # tremor oscillation frequency (2pi * hz)

    # TODO: Finish this function
    # Initialize randomized constants for simplified environment (acceleration, velocity, etc.)
    def init_constants_randomized(self):
        # spatial info - Information based on previous research from:
        # TODO: Find Real Values
        self.axi = random.uniform(-2, 2)
        self.ayi = random.uniform(-2, 2)
        self.azi = random.uniform(-2, 2)
        self.vxi = random.uniform(-3, 3)
        self.vyi = random.uniform(-3, 3)
        self.vzi = random.uniform(-3, 3)
        self.xi = random.uniform(-2, 2)
        self.yi = random.uniform(-2, 2)
        self.zi = random.uniform(-2, 2)

        # rotational info
        self.rolli = random.uniform(-2, 2)
        self.pitchi = random.uniform(-2, 2)
        self.yawi = random.uniform(-2, 2)
        self.vx_angular = random.uniform(-2, 2)
        self.vy_angular = random.uniform(-2, 2)
        self.vz_angular = random.uniform(-2, 2)

        # tremor info
        self.t_amp = random.uniform(0, 4)  # units of ___
        self.t_freq = 2 * math.pi * random.uniform(4, 12)  # tremor oscillation frequency (2pi * hz), range of 4-12 hz

# True data from the Environment, Serves as a ground truth
class SpatioTemporalData():
    def __init__(self):
        # Initialize spatial information
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.time = 0.0
        self.wrist_angle = 0.0
        self.pressure = 0.0
        self.t_amplitude = 0.0
        self.t_freq = 0.0

# Sensor Data as Metadata for the IMU
class SensorData():
    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.vx_angular = 0.0
        self.vy_angular = 0.0
        self.vz_angular = 0.0
        self.flex_angle = 0.0
        self.pressure_output = 0.0


    def integrate_update(self, movement, st_info, step):
        # approximate linear velocities
        self.vx = self.get_vx(st_info, step)
        self.vy = self.get_vy(st_info, step)
        self.vz = self.get_vz(st_info, step)

        # approximate linear accel
        self.ax = (self.get_vx(st_info, step+1) - self.get_vx(st_info, step-1))/(2*DELTA_T)
        self.ay = (self.get_vy(st_info, step+1) - self.get_vy(st_info, step-1))/(2*DELTA_T)
        self.az = (self.get_vz(st_info, step+1) - self.get_vz(st_info, step-1))/(2*DELTA_T)

        # approximate angular velocities
        self.vx_angular = (st_info[step+1].roll - st_info[step-1].roll)/(2*DELTA_T)
        self.vy_angular = (st_info[step+1].pitch - st_info[step-1].pitch)/(2*DELTA_T)
        self.vz_angular = (st_info[step+1].yaw - st_info[step-1].yaw)/(2*DELTA_T)

    def get_vx(self, st_info, step):
        return (st_info[step+1].x - st_info[step-1].x)/(2*DELTA_T)

    def get_vy(self, st_info, step):
        return (st_info[step+1].y - st_info[step-1].y)/(2*DELTA_T)

    def get_vz(self, st_info, step):
        return (st_info[step+1].z - st_info[step-1].z)/(2*DELTA_T)

    def reset(self):
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.vx_angular = 0.0
        self.vy_angular = 0.0
        self.vz_angular = 0.0
        self.flex_angle = 0.0
        self.pressure_output = 0.0