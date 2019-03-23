# LSTM for international airline passengers problem with regression framing
import sys
sys.path.append("F:/Apollo_NT/TremorWear_Env/")

import numpy as np
import scipy.signal as signal
from scipy import stats
import matplotlib.pyplot as plt
import math
import random
from scanf import scanf
from keras import regularizers
from keras.models import Sequential, model_from_json
from keras.layers import Dense, TimeDistributed
from keras.layers import CuDNNLSTM, Dropout, Activation, Conv1D, MaxPooling1D, LeakyReLU, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import CSVLogger
from environment.environment import TremorSim
from environment.DataGenerator import DataGenerator
from environment.DataGeneratorBatch import DataGeneratorBatch
from agents import preprocess


# normalize the dataset
def Keras_LSTM():
    np.random.seed(7)
    random.seed(7)

    input_size = 64
    output_size = 64
    batch_size = 10000 - input_size - output_size

    training_generator = DataGenerator(batch_size, input_size, output_size)
    validation_generator = DataGenerator(1000, input_size, output_size)

    # create and fit the LSTM network
    model = Sequential()

    # n_out = (n_in + 2p - k)/s + 1
    model.add(Conv1D(64, 3, padding='same', input_shape=(input_size, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2, stride=2, padding='same'))
    model.add((Dropout(rate=0.5)))

    model.add(Conv1D(128, 3, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2, stride=2, padding='same'))
    model.add(Dropout(rate=0.5))

    model.add(CuDNNLSTM(32))
    model.add(Dropout(rate=0.5))
    model.add((Dense(output_size)))
    model.add((Activation('linear')))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Load the weights
    #model.load_weights('model_weights.h5')

    # model.fit(trainX, trainY, epochs=1000, batch_size=512, verbose=2)
    filepath = "weights.best.hdf5"
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    checkpoint = ModelCheckpoint(filepath, monitor='mean_squared_error', verbose=1, save_best_only=False, mode='min')
    callbacks_list = [checkpoint, csv_logger]
    model.fit_generator(generator=training_generator, validation_data=validation_generator, callbacks=callbacks_list,
                        verbose=1, epochs=3000)

    # Save the weights
    model.save_weights('model_weights.h5')

    # Save the model architecture
    with open('model_architecture.json', 'w') as f:
        f.write(model.to_json())

def Keras_LSTM_Val():
    np.random.seed(7)
    random.seed(7)

    input_size = 64
    output_size = 64
    processor = preprocess.SignalProcessor(500)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # create and fit the LSTM network
    model = Sequential()

    # n_out = (n_in + 2p - k)/s + 1
    model.add(Conv1D(64, 3, input_shape=(input_size, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2, stride=2, padding='same'))
    model.add(Dropout(0.4))

    model.add(Conv1D(32, 3, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(pool_size=2, stride=2, padding='same'))
    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(CuDNNLSTM(16))
    model.add(Dense(output_size))
    model.add(Activation('linear'))

    # Load the weights
    model.load_weights('model_weights.h5')
    # model.compile(loss='mean_squared_error', optimizer='adam')
    gsum = 0
    diff_sum = 0
    z_sum = 0
    trials = 1000
    for j in range(trials):
        #testX, testY, ys = get_seq(input_size, output_size, processor, "Sarah_Resting")
        testX, testY, ys = generate_seq(input_size, output_size, processor)
        predictY = model.predict(testX)

        predict = []
        groundp = []

        sum = 0
        diff = 0
        for i in range(len(predictY)):
            predict.append(predictY[i][4])
            groundp.append(testY[i][4])
            sum += math.fabs(predictY[i][4] - testY[i][4])

        p2, _ = signal.find_peaks(groundp, prominence=3)
        z1 = get_zeros(predict)
        z2 = get_zeros(groundp)
        m = min(len(z1), len(z2))

        z = 0
        for i in range(m):
            z += math.fabs(z1[i] - z2[i])
        if z / m > 0.01:
            z_sum += z_sum/(trials-1)
        else:
            z_sum += z / m
        print("Average Prediction Phase Error: {}".format(z/m))

        for i in range(len(p2)):
            diff += math.fabs((groundp[p2[i]]-predict[p2[i]])/groundp[p2[i]])
        if len(p2) == 0:
            diff = diff_sum/(trials-1)
        else:
            diff = diff / len(p2)
        diff_sum += diff
        print("Average Peak Amplitude Error: {}".format(diff))

        plt.plot(predict[0:1001], label='Predicted Tremor-Induced Motion by Inferencing [50ms] Ahead')
        plt.xlabel('Time [ms]', fontsize=16)
        plt.ylabel('Amplitude [rad/s]', fontsize=16)
        plt.plot(groundp[0:1001], label='True Tremor-Induced Motion')
        plt.legend(loc='upper left')
        plt.show()

    print("Total Average Average Prediction Phase Error over {} Runs: {}".format(trials, z_sum/trials))
    print("Total Average Peak Amplitude Error over {} Runs: {}".format(trials, diff_sum / trials))

def create_dataset(dataset, gdataset, input_size=1, output_size=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-(output_size+input_size-1)-1):
        a = dataset[i:(i+input_size), 0]
        b = gdataset[(i+input_size):(i+input_size+output_size), 0]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


def generate_seq(input_size, output_size, processor):
    env = TremorSim(1000)
    ground, data = env.generate_sim()
    x = np.arange(0, 1000)
    y = np.arange(0, 1000)

    xs = stats.zscore([data[i].getGyroReading() for i in x])
    ys = stats.zscore([ground[i].getTremor() for i in y])

    plt.xlabel('Time [ms]', fontsize=10)
    plt.ylabel('Amplitude [rad/s]', fontsize=10)
    plt.plot(xs)
    plt.show()

    filt_x, _ = processor.Bandpass_Filter(xs, 3, 13, 5)

    dataset = np.reshape(filt_x, [-1, 1])
    gdataset = np.reshape(ys, [-1, 1])

    # gdataset = scaler.fit_transform(gdataset)
    # dataset = scaler.transform(dataset)

    # reshape into X=t and Y=t+1
    testX, testY = create_dataset(dataset, gdataset, input_size, output_size)

    # reshape input to be [samples, time steps, features]
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    return testX, testY, ys


def get_seq(input_size, output_size, processor, file):
    _, _, _, _, _, gy, _ = read(file)

    ys = stats.zscore(gy)

    filt, _ = processor.Bandpass_Filter(ys, 3, 13, 5)

    dataset = np.reshape(filt, [-1, 1])
    gdataset = np.reshape(filt, [-1, 1])

    # reshape into X=t and Y=t+1
    testX, testY = create_dataset(dataset, gdataset, input_size, output_size)

    # reshape input to be [samples, time steps, features]
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    return testX, testY, ys


def read(filename):
    ax, ay, az, gx, gy, gz = [], [], [], [], [], []

    with open("../env_movements/" + filename + ".txt") as data:
        freq = scanf("%f", data.readline())
        for line in data:
            # time[ms], ax, ay, az, gx, gy, gz
            _, axt, ayt, azt, gxt, gyt, gzt = scanf("%f %f %f %f %f %f %f", line)
            ax.append(axt)
            ay.append(ayt)
            az.append(azt)
            gx.append(gxt)
            gy.append(gyt)
            gz.append(gzt)

    return freq, ax, ay, az, gx, gy, gz

def get_zeros(seq):
    zeros = []
    prev_value = seq[0]
    for i in range(len(seq)-1):
        if seq[i] == 0:
            zeros.append(i/500.0)
        elif prev_value > 0 and seq[i] < 0:
            zeros.append(prev_value / (500 * (prev_value - seq[i])) + (i-1) / 500.0)
        elif prev_value < 0 and seq[i] > 0:
            zeros.append(prev_value / (500 * (prev_value - seq[i])) + (i-1) / 500.0)
        prev_value = seq[i]
    return zeros

def Voluntary_Index():
    for i in range(100):
        pass

if __name__ == '__main__':
    Keras_LSTM()

