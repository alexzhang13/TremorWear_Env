import numpy as np
import keras
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from environment.environment import TremorSim
from agents import preprocess

class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=32, INPUT_SIZE=64, OUTPUT_SIZE=64, shuffle=True):
        self.INPUT_SIZE = INPUT_SIZE
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.processor = preprocess.SignalProcessor(500)
        self.train_size = ((10000-INPUT_SIZE-OUTPUT_SIZE)//batch_size)
        self.max_step = self.train_size*batch_size
        self.env = TremorSim(10000)
        self.trainX, self.trainY = None, None

        # Initialize First Epoch
        self.on_epoch_end()

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y = self.__data_generation(index)
        X = np.reshape(X, (X.shape[1], X.shape[0], 1))

        return X, y

    def __data_generation(self, idx):
        return np.array(self.trainX[idx]), np.expand_dims(self.trainY[idx], axis=0)

    def get_sequence(self):
        ground, data = self.env.generate_sim()

        x = np.arange(0, 10000)
        y = np.arange(0, 10000)

        xs = stats.zscore([data[i].getGyroReading() for i in x])
        ys = stats.zscore([ground[i].getTremor() for i in y])

        # Bandpass Filter
        filt_x, _ = self.processor.Bandpass_Filter(xs, 3, 13, 5)

        # Normalize between 0 and 1 for LSTM [|Y| always > than |X|]
        dataset = np.reshape(filt_x, [-1, 1])
        gdataset = np.reshape(ys, [-1, 1])

        return [dataset, gdataset]


    def create_dataset(self, dataset, gdataset, input_size=1, output_size=1):
        const = np.random.randint(0, len(dataset) - (output_size + input_size + self.batch_size)-1)
        dataX, dataY = [], []
        for i in range(self.batch_size):
            a = dataset[i+const:(i + const + input_size), 0]
            b = gdataset[(i + const + input_size):(i + const + input_size + output_size), 0]
            dataX.append(a)
            dataY.append(b)
        return np.array(dataX), np.array(dataY)

    def __len__(self):
        return self.batch_size

    def on_epoch_end(self):
        seq, gt = self.get_sequence()
        trainX, trainY = self.create_dataset(seq, gt, self.INPUT_SIZE, self.OUTPUT_SIZE)
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        self.trainX, self.trainY = trainX, trainY