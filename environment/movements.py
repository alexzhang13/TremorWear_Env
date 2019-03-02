import numpy as np
from scanf import scanf

AMP_STD = 0.005
FREQ_STD = 0.01
PHASE_STD = 0.005


class BaseMovement():
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2, path=""):
        self.v_angular = v_angular
        self.amp1 = amp1
        self.freq1 = freq1
        self.phase1 = phase1
        self.amp2 = amp2
        self.freq2 = freq2
        self.phase2 = phase2
        # self.chooser = np.random.randint(0, 2) # 0 = roll, 1 = pitch, 2 = yaw

        # self.seq = self.read(path)[self.chooser]

    def angular_transform(self, step):
        return self.v_angular

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
        ax, ay, az, gx, gy, gz = [], [], [], [], [], []

        with open(filename) as data:
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
        super(MovementResting, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2)

    def angular_transform(self, step):
        return 0.0


# Eating Movement (Spoon and a Bowl)
class MovementEating(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementEating, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2)

    def angular_transform(self, step):
        return 0.0


# Rotating Hand Movement (Moving Slowly From Side to Side)
class MovementRotation(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementRotation, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2)

    def angular_transform(self, step):
        return 0.0


# Writing Movement
class MovementWriting(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementWriting, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2)

    def angular_transform(self, step):
        return 0.0


# Drinking Movement
class MovementDrinking(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementDrinking, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2)

    def angular_transform(self, step):
        return 0.0


# Lifting Object with Hand
class MovementLifting(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(MovementLifting, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2)

    def angular_transform(self, step):
        return 0.0