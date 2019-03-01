import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import random as rand

colordict = {
    0: 'b',
    1: 'g',
    2: 'r',
    3: 'c',
    4: 'm',
    5: 'y',
    6: 'k',
}

# Process 3 sequences of spatial information (gx, gy, gz)
class SignalProcessor():
    def __init__(self, sample_rate):
        self.srate = sample_rate

    # Simple Truncating Filter
    def SimpleFilter(self, sequence, low_freq, high_freq):
        # Convert to FFT
        fourier, freq = self.Fourier(sequence)

        # Filter
        for i in range(len(freq)):
            if freq[i] < low_freq or freq[i] > high_freq:
                fourier[i] = 0

        self.SaveFFTGraph(fourier, freq, "Filtered")

        # Return iFFT of filtered FFT
        return self.IFourier(fourier, len(sequence))

    # Simple Bandpass Filter
    def Bandpass_Filter(self, sequence, low_freq, high_freq, order):
        nyq = 0.5*self.srate # Nyquist Frequency
        low = low_freq / nyq
        high = high_freq / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered_sequence = filtfilt(b, a, sequence)
        time_window = np.linspace(0, len(sequence) * (1 / self.srate), len(sequence))
        return filtered_sequence, time_window

    def Fourier(self, spatial_sequence):
        fourier = np.fft.fft(spatial_sequence)
        freq = np.linspace(0.0, 0.5*self.srate, len(spatial_sequence)/2)
        return fourier, freq

    def IFourier(self, fourier_sequence, window_size):
        ifourier = np.fft.ifft(fourier_sequence)
        time_window = np.linspace(0, window_size*(1/self.srate), window_size)

        return ifourier, time_window

    def Bandpass_All(self, sequences, low_freq, high_freq, order):
        filtered_sequences = []
        for i in range(len(sequences)):
            filtered_sequences.append((self.Bandpass_Filter(sequences[i], low_freq, high_freq, order)))

        return filtered_sequences

# -------------------------------- TESTER FUNCTIONS -------------------------------- #

    def SaveIFFTGraph(self, ifourier, window, name):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(" Composed Graph: " + name, fontsize=18)
        ax.set_ylabel("Angular V(t) [rad/s]")
        ax.set_xlabel("Time [s]")

        # ax.plot(freq, 2.0/len(fourier) * np.abs(fourier[:len(fourier)//2]))
        ax.plot(window, ifourier.real)
        ax.plot(window, ifourier.imag)

        plt.legend(['real', 'imaginary'], loc='upper left')
        plt.show()

        fig.savefig("../img/" + name + ".png")

    def SaveFFTGraph(self, fourier, freq, name):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(" FFT Graph: " + name, fontsize=18)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Frequency [Hz]")

        ax.plot(freq, 2.0/len(fourier) * np.abs(fourier[:len(fourier)//2]))
        plt.xlim(0, 50)

        plt.show()

        fig.savefig("../img/" + name + ".png")

    def SaveButterFilterGraph(self, y, window, name):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(" Filtered Graph: " + name, fontsize=18)
        ax.set_ylabel("Angular V(t) [rad/s]")
        ax.set_xlabel("Time [s]")

        # ax.plot(freq, 2.0/len(fourier) * np.abs(fourier[:len(fourier)//2]))
        ax.plot(window, y)
        plt.show()

        fig.savefig("../img/" + name + ".png")

    def FilterTest(self, sequence, name):
        filtered, window = self.Bandpass_Filter(sequence, 3, 13, 5)
        self.SaveButterFilterGraph(filtered, window, name + "_Filtered")

        fourier, freq = self.Fourier(sequence)
        self.SaveFFTGraph(fourier, freq, name + ": Original FFT")

        fourier2, freq2 = self.Fourier(filtered)
        self.SaveFFTGraph(fourier2, freq2, name + ": Filtered FFT")

        voluntary = np.subtract(sequence, filtered)
        self.SaveButterFilterGraph(voluntary, window, name + "_VoluntaryMotion")

        fourier3, freq3 = self.Fourier(voluntary)
        self.SaveFFTGraph(fourier3, freq3, name + ": Voluntary FFT")

    def FourierTest(self, sequence, name):
        fourier, freq = self.Fourier(sequence)
        self.SaveFFTGraph(fourier, freq, name + "_FFT")

        ifourier, window = self.IFourier(fourier, len(sequence))
        self.SaveIFFTGraph(ifourier, window, name + "_C")

    def WindowFourier(self, sequence, window_size, name):
        fig = plt.figure(figsize=(8,4))

        plt.ion()
        plt.show()

        for i in range(0, len(sequence)-window_size, 1):
            filtered, window = self.Bandpass_Filter(sequence[i:window_size+i], 3, 13, 5)
            fourier, freq = self.Fourier(filtered)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(" Shifting Window: " + name, fontsize=18)
            ax.set_ylabel("Amplitude")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 0.02)
            ax.plot(freq, 2.0 / len(fourier) * np.abs(fourier[:len(fourier)//2]), 'bo-', color=colordict[0])
            plt.draw()
            plt.pause(0.1)
            plt.clf()

        plt.waitforbuttonpress()

    def SaveSequences(self, gx, gy, gz, filename):
        xfiltered, window = self.Bandpass_Filter(gx, 3, 13, 5)
        yfiltered, _ = self.Bandpass_Filter(gy, 3, 13, 5)
        zfiltered, _ = self.Bandpass_Filter(gz, 3, 13, 5)

        xvoluntary = np.subtract(gx, xfiltered)
        yvoluntary = np.subtract(gy, yfiltered)
        zvoluntary = np.subtract(gz, zfiltered)

        file1 = open("../data/" + filename + "_Filtered.txt", 'x')
        file1.write("500\n")
        for i in range(len(window)):
            file1.write("{} {} {} {} {} {} {}\n".format(window[i], 0.0, 0.0, 0.0, xfiltered[i], yfiltered[i], zfiltered[i]))
        file1.close()

        file2 = open("../data/" + filename + "_Voluntary.txt", 'x')
        file2.write("500\n")
        for i in range(len(window)):
            file2.write("{} {} {} {} {} {} {}\n".format(window[i], 0.0, 0.0, 0.0, xvoluntary[i], yvoluntary[i], zvoluntary[i]))
        file2.close()
