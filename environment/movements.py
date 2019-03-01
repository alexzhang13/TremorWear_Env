import numpy as np
import copy

AMP_STD = 0.005
FREQ_STD = 0.01
PHASE_STD = 0.005


class BaseMovement():
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        self.v_angular = v_angular
        self.amp1 = amp1
        self.freq1 = freq1
        self.phase1 = phase1
        self.amp2 = amp2
        self.freq2 = freq2
        self.phase2 = phase2

    def angular_transform(self, t):
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


# Case 1, Resting Hand Movement
class Resting(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(Resting, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2)

    def angular_transform(self, t):
        return 0.0


# Case 2, Free Hand Rotational Movement (Constant)
class FreeHandConstantRotation(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(FreeHandConstantRotation, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2)

    def angular_transform(self, t):
        return 0.0

    def yaw(self, vz_angular, yawi, t):
        return yawi + vz_angular*t


# Case 3, Free Hand Rotational Movement
class FreeHandRotation(BaseMovement):
    def __init__(self, v_angular, amp1, freq1, phase1, amp2, freq2, phase2):
        super(FreeHandRotation, self).__init__(v_angular, amp1, freq1, phase1, amp2, freq2, phase2)

    def angular_transform(self, t):
        return 0.0

    def yaw(self, vz_angular, yawi, t):
        return yawi + vz_angular*t