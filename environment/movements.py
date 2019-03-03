import os
import numpy as np
from scanf import scanf

AMP_STD = 0.005
FREQ_STD = 0.01
PHASE_STD = 0.005
ANGULAR_STD = 0.005

class BaseMovement():
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2, path=""):
        self.v_angular = v_angular
        self.amp1 = amp1
        self.freq1 = freq1
        self.phase1 = phase1
        self.amp2 = amp2
        self.freq2 = freq2
        self.phase2 = phase2
        self.chooser = np.random.randint(0, 2) # 0 = roll, 1 = pitch, 2 = yaw
        self.multiplier = np.random.normal(1.0, 0.1)

        self.seq = self.read(path)[self.chooser]

    def angular_transform(self, step, stepi):
        return self.seq[step+stepi] * self.multiplier + np.random.normal(0.0, ANGULAR_STD)

    def tremor_transform(self, t):
        return self.amp1 * np.sin(self.freq1 *2*np.pi* t + self.phase1) + \
               self.amp2 * np.sin(self.freq2 *2*np.pi* t + self.phase2)

    def updateTremor(self):
        self.amp1 = np.random.normal(self.amp1, AMP_STD, None)
        self.amp2 = np.random.normal(self.amp2, AMP_STD, None)

        self.freq1 = np.random.normal(self.freq1, FREQ_STD, None)
        self.freq2 = np.random.normal(self.freq2, FREQ_STD, None)

        self.phase1 = np.random.normal(self.phase1, PHASE_STD, None)
        self.phase2 = np.random.normal(self.phase2, PHASE_STD, None)

        return self.amp1, self.freq1, self.phase1, self.amp2, self.freq2, self.phase2

    @staticmethod
    def read(filename):
        gx, gy, gz = [], [], []
        with open("./env_movements/" + filename + ".txt") as data:
            freq = scanf("%f", data.readline())
            for line in data:
                _, _, _, _, gxt, gyt, gzt = scanf("%f %f %f %f %f %f %f", line)
                gx.append(gxt)
                gy.append(gyt)
                gz.append(gzt)

        return gx, gy, gz

# Resting Motion
class MovementResting(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementResting, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2, "MovementResting")


# Eating Movement (Spoon and a Bowl)
class MovementEating(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementEating, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2, "MovementEating")


# Rotating Hand Movement (Moving Slowly From Side to Side)
class MovementRotation(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementRotation, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2, "MovementRotation")


# Writing Movement
class MovementWriting(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementWriting, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2, "MovementWriting")


# Drinking Movement
class MovementDrinking(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementDrinking, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2, "MovementDrinking")


# Rotating Wrist in Certain Patterns
class MovementWrist(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementWrist, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2, "MovementWrist")


# Grabbing Object
class MovementGrab(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementGrab, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2, "MovementGrab")