#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from future.builtins import range  # pylint: disable=redefined-builtin
from itertools import tee

from environment.environment import TremorSim
from agents.lstm_agent import LSTM_Agent
from agents.preprocess import SignalProcessor

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import logging
import datetime
import random

SAMPLE_RATE = 500 # in hz
TRAINING_STEPS = 2000000
TIME_STEPS = 300
INPUT_SIZE = 100
OUTPUT_SIZE = 100
CELL_SIZE = 4
NUM_LAYERS = 1
LR = 0.0005
KEEP_PROB = 0.5
DROPOUT_IN = 0.5

parser = argparse.ArgumentParser(description='Main for running agents in the Tremor Environment')

# QAgent Arguments
parser.add_argument("--network", type=str, default="LSTM", help="Network to Run")
parser.add_argument("--batch_size", type=int, default=256, help="Batch Size for Training")
parser.add_argument("--save_model_folder", type=str, default="saved_models/model_", help="Path to saved model")
parser.add_argument("--load_model_folder", type=str, default="saved_models/checkpoint", help="Path to saved model")
parser.add_argument("--real", dest="real_data", action="store_true", help="Use Patient Data for Training")
parser.add_argument("--simulation", dest="real_data", action="store_false", help="Use Simulation for Training")
parser.set_defaults(real_data=False)
parser.add_argument("--train", dest="is_training", action="store_true", help="Train Neural Network")
parser.add_argument("--test", dest="is_training", action="store_false", help="Test Neural Network")
parser.set_defaults(is_training=True)
parser.add_argument("--save_model", dest="save_model", action="store_true", help="Save LSTM Data")
parser.set_defaults(save_model=False)
parser.add_argument("--load_model", dest="load_model", action="store_true", help="Load LSTM Data")
parser.set_defaults(load_model=False)
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug to Log")
parser.set_defaults(debug=False)
parser.add_argument("--graph", dest="graph", action="store_true", help="Graph Output")
parser.set_defaults(graph=False)

args = parser.parse_args()

def main():
    if args.real_data is False:
        print("Using Simulation Data")
    else:
        print("Using Real Data")

    if args.network == "LSTM":
        LSTM()
    if args.network == "Keras_LSTM":
        pass
    if args.network == "Test":
        Test()

def LSTM():
    # Initialize Seeds
    np.random.seed(7)
    random.seed(7)

    # Initialize LSTM
    model = LSTM_Agent(args.is_training, LR, NUM_LAYERS, TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, args.batch_size, KEEP_PROB,
                       DROPOUT_IN)

    if args.debug is True:
        print("Debugging to Log File...")
        now = datetime.datetime.now()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename="./logs/" + "log_{}.log".format(now.strftime("%m-%d-%Y-%H-%M-%S")), level=logging.INFO)

    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tf_logs", sess.graph)
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=1000)

    env = TremorSim(INPUT_SIZE + TIME_STEPS + OUTPUT_SIZE)
    scaler = MinMaxScaler(feature_range=(-1,1))

    if args.load_model is True:
        load_model(saver, sess, args.load_model_folder)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    plt.ion()
    for episode in range(TRAINING_STEPS):
        xs, ys = get_batch_sequence(args.batch_size, env, scaler)
        feed_dict = {
            model.x: xs,
            model.y: ys,
        }

        _, cost, state, pred = sess.run(
            [model.train_op, model.loss, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
        logging.info('Episode: {}, Loss: {}'.format(episode, round(cost, 4)))
        if episode % 10 == 0:
            print('Episode: {}, Loss: {}\n'.format(episode, round(cost, 4)))
        result = sess.run(merged, feed_dict)
        writer.add_summary(result, episode)

        if args.save_model is True and episode % 100 == 0:
            save_model(saver, sess, args.save_model_folder, episode//100)
        if args.graph is True and episode % 10 == 0:
            # plotting
            plt.plot(np.arange(0, OUTPUT_SIZE*2, 2), ys[0][0], 'r', np.arange(0, OUTPUT_SIZE*2, 2), pred[0][0:OUTPUT_SIZE], 'b--')
            plt.ylim((-1, 1))
            plt.show()
            plt.pause(5)
            plt.close()


def Test():
    env = TremorSim(200)
    ground, data = env.generate_sim()
    processor = SignalProcessor(500)

    fig = plt.figure(figsize=(8.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(" Generated Simulation Data ", fontsize=18)
    ax.set_ylabel("Gyro: [rad/s]")
    ax.set_xlabel("Time [ms]")

    values = [x.getGyroReading() for x in data]
    gvalues = [y.getTremor() for y in ground]
    filtered, freq = processor.Bandpass_Filter(values, 3, 13, 5)

    plt.plot(np.arange(0, 400, 2), values)
    plt.plot(np.arange(0, 400, 2), filtered)
    plt.plot(np.arange(0, 400, 2), gvalues)
    plt.legend(['sensor(unfiltered)', 'sensor(filtered)', 'ground'], loc='upper left')

    plt.show()

    processor = SignalProcessor(500.0)

    fourier, freq = processor.Fourier(values)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(" FFT Graph: ", fontsize=18)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Frequency [Hz]")

    ax.plot(freq, 2.0 / len(fourier) * np.abs(fourier[:len(fourier) // 2]))
    plt.xlim(0, 50)

    plt.show()


def get_experimental_sequence(env):
    global BATCH_START

    seq, gt = [], []
    x = np.arange(0, TIME_STEPS+INPUT_SIZE-1)
    y = np.arange(INPUT_SIZE, TIME_STEPS+INPUT_SIZE+OUTPUT_SIZE-1)
    xs = 2*np.sin(2*x)+2
    ys = 2*np.sin(2*y)+2

    for each in window(xs, INPUT_SIZE):
        seq.append(each)
    for each in window(ys, OUTPUT_SIZE):
        gt.append(each)

    return [seq, gt]


def get_sequence(env, scaler):
    seq, gt = [], []
    ground, data = env.generate_sim()

    x = np.arange(0, TIME_STEPS+INPUT_SIZE-1)
    y = np.arange(INPUT_SIZE, TIME_STEPS+INPUT_SIZE+OUTPUT_SIZE-1)

    xs = [data[i].getGyroReading() for i in x]
    ys = [ground[i].getTremor() for i in y]

    # Bandpass Filter
    processor = SignalProcessor(SAMPLE_RATE)
    x_filtered, _ = processor.Bandpass_Filter(xs, 3, 13, 5)

    # Normalize between 0 and 1 for LSTM [|Y| always > than |X|]
    y_values = np.asarray(ys).reshape((len(ys), 1))
    y_values = scaler.fit_transform(y_values)
    y_values = y_values.reshape(1, len(ys))

    x_values = x_filtered.reshape((len(x_filtered), 1))
    x_values = scaler.transform(x_values)
    x_values = x_values.reshape(1, len(x_values))

    for each in window(x_values[0], INPUT_SIZE):
        seq.append(each)
    for each in window(y_values[0], OUTPUT_SIZE):
        gt.append(each)

    # a = np.asarray(seq)
    # a.transpose()
    # a = scaler.transform(a)
    # a.transpose()

    return [seq, gt]

def get_batch_sequence(batch_size, env, scaler):
    xs, ys = [], []
    for i in range(batch_size):
        seq, gt = get_sequence(env, scaler)
        xs.append(seq)
        ys.append(gt)
    return xs, ys

def save_model(saver, sess, path, count):
    path += "{}".format(args.network)
    spath = saver.save(sess, "saved_models/model.cpkt")
    logging.info("Model Saved at Path: ".format(spath))


def load_model(saver, sess, path):
    saver = tf.train.import_meta_graph('/tmp/model.ckpt.meta')
    saver.restore(sess, "saved_models/model.cpkt")
    logging.info("Model Loaded from Path: {}".format(path))


def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)

if __name__ == "__main__":
    main()
