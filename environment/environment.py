# Simulation Environment for Non-Reinforcement Learning Neural Networks

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from environment import movements

import importlib

# Constants
IMU_UPPER_BOUND = 16384
IMU_LOWER_BOUND = -16384
LOCKING_UPPER_BOUND = 1.0
LOCKING_LOWER_BOUND = 0.0
DEVICE_HZ = 500.0  # Operations per second
DELTA_T = 1 / DEVICE_HZ  # Change in time per episode
DEG2RAD = 3.1415926535 / 180.0
RAD2DEG = 180.0 / 3.1415926535
NOISE_STD = 0.005

simulated_action_space = [
    "MovementResting",
    "MovementEating",
    "MovementRotation",
    "MovementWriting",
    "MovementDrinking",
    "MovementWrist",
    "MovementGrab"
]

class TremorSim():
    def __init__(self, max_steps):
        # Initialize Spatiotemporal information from start to end of episode [ground truth]
        self.init_constants()
        self.steps = 0
        self.max_steps = max_steps

        self.v_angular = 0.0

        # tremor info
        self.amp1 = 0.0
        self.freq1 = 0.0
        self.phase1 = 0.0
        self.amp2 = 0.0
        self.freq2 = 0.0
        self.phase2 = 0.0

        np.random.seed(7)
        random.seed(7)

    # Generate Full Simulation Case
    def generate_sim(self):
        self._simulated_tremor = self.init_movement()

        # Generate Ground Truth Sequence (Spatial "real"[simulated] world information)
        st_info = []

        # Generate Global Data (Tremor Motion is Added Afterwards)
        for i in range(self.max_steps):
            spatiotemporal = self.generate_stdata(self._simulated_tremor, i)
            st_info.append(spatiotemporal)

        # Generate Window of Sensor Data from Real World Data
        sensor_info = []
        for i in range(self.max_steps):
            sensordata = SensorData(st_info, i)
            sensor_info.append(sensordata)

        return st_info, sensor_info

    # Initialize random movement at the start of each episode
    def init_movement(self):
        # Randomly Initialize Constants
        self.init_constants()

        # Initialize Simulated Env (Pick a case randomly)
        i = random.randint(0, 6)  # random.randint produces a number between a,b INCLUSIVE (as opposed to np)
        env = __import__("environment")
        movements = getattr(env, "movements")
        _class = getattr(movements, simulated_action_space[i])
        _instance = _class(self.v_angular, self.amp1, self.freq1, self.phase1, self.amp2, self.freq2, self.phase2)

        return _instance

    def choose_movement(self, i):
        # Randomly Initialize Constants
        self.init_constants()

        # Initialize Simulated Env (Pick a case randomly)
        env = __import__("environment")
        movements = getattr(env, "movements")
        _class = getattr(movements, simulated_action_space[max(0, min(i, 6))])
        _instance = _class(self.v_angular, self.amp1, self.freq1, self.phase1, self.amp2, self.freq2, self.phase2)

        return _instance

    # Initialize randomized constants for simplified environment (acceleration, velocity, etc.)
    def init_constants(self):
        # Rotational Info
        self.v_angular = random.uniform(-2, 2)

        # Tremor Info
        self.amp1 = random.uniform(0.0, 1.0)
        self.freq1 = random.uniform(3, 12)
        self.phase1 = random.uniform(0, 2*np.pi)

        # Chance for Second Tremor
        choice = random.randint(0, 1)
        if choice == 0:
            self.amp2 = 0.0
            self.freq2 = 0.0
            self.phase2 = 0.0
        else:
            self.amp2 = random.uniform(0.0, 1.0)
            self.freq2 = random.uniform(3, 12)
            self.phase2 = random.uniform(0, 2 * np.pi)

    # Tester Function for Cases
    def init_defined_constants(self):
        self.v_angular = 0.0

        # tremor info
        self.amp1 = 0.0
        self.freq1 = 0.0
        self.phase1 = 0.0
        self.amp2 = 0.0
        self.freq2 = 0.0
        self.phase2 = 0.0

    # Update sensor info at next time step (generated simulation) TODO: Update for New Movement Code (Local to Global)
    def generate_stdata(self, movement, step):
        next_st_data = SpatioTemporalData()

        # Update Voluntary Motion
        next_st_data.v_angular = movement.angular_transform(step)
        next_st_data.time = step * DELTA_T

        # Update Tremor Data (Not Added Until After)
        next_st_data.amp1, next_st_data.freq1, next_st_data.phase1, next_st_data.amp2, next_st_data.freq2, next_st_data.phase2 = movement.updateTremor()
        next_st_data.t_angular = movement.tremor_transform(step * DELTA_T)

        return next_st_data

    def setLength(self, length):
        self.max_steps = length

# True data from the Environment, Serves as a ground truth
class SpatioTemporalData():
    def __init__(self):
        # Initialize spatial information
        self.v_angular = 0.0
        self.t_angular = 0.0
        self.time = 0.0
        self.wrist_angle = 0.0

        # Tremor Freq Info
        self.amp1 = 0.0
        self.freq1 = 0.0
        self.phase1 = 0.0
        self.amp2 = 0.0
        self.freq2 = 0.0
        self.phase2 = 0.0

    def setTremorInfo(self, amp1, freq1, phase1, amp2, freq2, phase2):
        self.amp1 = amp1
        self.freq1 = freq1
        self.phase1 = phase1
        self.amp2 = amp2
        self.freq2 = freq2
        self.phase2 = phase2

    def getTremor(self):
        return self.t_angular

    def getAngularV(self):
        return self.v_angular

# Sensor Data as Metadata for the IMU
class SensorData():
    def __init__(self, st_info, step):
        # Estimate Gyroscope Readings with Added Tremor Motion
        self.gyro = st_info[step].v_angular + st_info[step].amp1 * \
        np.sin(st_info[step].freq1*2*np.pi*st_info[step].time + st_info[step].phase1) + st_info[step].amp2 * \
        np.sin(st_info[step].freq2*2*np.pi*st_info[step].time + st_info[step].phase2) + self.noise()

        self.flex_angle = 0.0

    def getGyroReading(self):
        return self.gyro

    @staticmethod
    def rotangle(an_x, an_y, an_z):
        ax = an_x * DEG2RAD
        ay = an_y * DEG2RAD
        az = an_z * DEG2RAD
        nv = [0, 0, 1]

        Rx = [[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]]
        Ry = [[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]]
        Rz = [[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]]

        R = np.matmul(np.matmul(Rz, Ry), Rx)
        rv = R * np.transpose(nv)

        return R, rv

    @staticmethod
    def plot_rotangle(R, rv):
        [x,y] = np.meshgrid(np.linspace(-4.0, 4.0, num=81), np.linspace(-4.0, 4.0, num=81))
        X = R[0][0]*x + R[0][1]*y
        Y = R[1][0]*x + R[1][1]*y
        Z = R[2][0]*x + R[2][1]*y

        if not plt.fignum_exists(0):
            plt.ion()
            plt.show()
        fig = plt.figure(0)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)
        ax.plot_surface(X=X, Y=Y, Z=Z)

        plt.draw()
        plt.pause(0.01)
        plt.clf()

        plt.quiver(0,0,0, rv[0][0], rv[1][0], rv[2][0])

    @staticmethod
    def noise():
        return np.random.normal(0, NOISE_STD, None)

