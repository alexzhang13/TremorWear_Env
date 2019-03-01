#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from future.builtins import range  # pylint: disable=redefined-builtin
from itertools import tee

from environment.environment import TremorSim
from agents.lstm_agent import LSTM_Agent
from agents.preprocess import SignalProcessor

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import logging
import datetime

SAMPLE_RATE = 500 # in hz
TRAINING_STEPS = 2000
TIME_STEPS = 500
BATCH_SIZE = 1
INPUT_SIZE = 128
OUTPUT_SIZE = 128
CELL_SIZE = 128
NUM_LAYERS = 1
LR = 0.006
KEEP_PROB = 0.75
DROPOUT_IN = 0.2

parser = argparse.ArgumentParser(description='Main for running agents in the Tremor Environment')

# QAgent Arguments
parser.add_argument("--network", type=str, default="LSTM", help="Network to Run")
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
parser.add_argument("--save_model_folder", type=str, default="saved_models/model_", help="Path to saved model")
parser.add_argument("--load_model_folder", type=str, default="saved_models/model_1", help="Path to saved model")
parser.add_argument("--debug", dest="debug", action="store_true", help="Debug to Log")
parser.set_defaults(debug=False)

args = parser.parse_args()

def main():
    if args.real_data is False:
        print("Using Simulation Data")
    else:
        print("Using Real Data")

    if args.network == "LSTM":
        LSTM()
    if args.network == "Test":
        env = TremorSim(1000)
        ground, data = env.generate_sim()

        fig = plt.figure(figsize=(8.0, 4.0))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(" Generated Simulation Data ", fontsize=18)
        ax.set_ylabel("Gyro: [rad/s]")
        ax.set_xlabel("Time [ms]")

        values = [x.getGyroReading() for x in data]

        plt.plot(np.arange(0, 2000, 2), values)
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

def LSTM():
    model = LSTM_Agent(args.is_training, LR, NUM_LAYERS, TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, KEEP_PROB,
                       DROPOUT_IN)
    env = TremorSim(TIME_STEPS + INPUT_SIZE + OUTPUT_SIZE)

    if args.debug is True:
        print("Debugging to Log File...")
        now = datetime.datetime.now()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename="/log/" + "log_{}.txt".format(now.isoformat()), level=logging.DEBUG)

    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=1000)

    if args.load_model is True:
        load_model(saver, sess, args.load_model_folder)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    plt.ion()
    plt.show()
    for i in range(TRAINING_STEPS):
        seq, gt, ys = get_sequence(env)
        feed_dict = {
            model.x: seq,
            model.y: gt,
        }

        _, cost, state, pred = sess.run(
            [model.train_op, model.loss, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
        if i % 20 == 0:
            logging.debug('loss: ', round(cost, 4))
            print('loss: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
        if args.save_model is True and i % 1000 == 0:
            save_model(saver, sess, args.save_model_folder, i//1000)
        if i > TRAINING_STEPS * (3/4):
            # plotting
            plt.plot(ys[0:OUTPUT_SIZE], gt[0], 'r', ys[0:OUTPUT_SIZE], pred[0], 'b--')
            plt.ylim((-4, 4))
            plt.draw()
            plt.pause(0.3)
            plt.clf()

def get_experimental_sequence(env):
    global BATCH_START

    seq, res = [], []
    x = np.arange(BATCH_START, BATCH_START+TIME_STEPS+INPUT_SIZE-1) / (8 * np.pi)
    y = np.arange(BATCH_START+INPUT_SIZE, BATCH_START+TIME_STEPS+INPUT_SIZE+OUTPUT_SIZE-1) / (8 * np.pi)
    xs = 2*np.sin(2*x)+2
    ys = 2*np.sin(2*x)+2
    for each in window(xs, INPUT_SIZE):
        seq.append(each)
    for each in window(ys, OUTPUT_SIZE):
        res.append(each)

    BATCH_START += TIME_STEPS
    return [seq, res, y]

def get_sequence(env):
    seq, gt = [], []
    ground, data = env.generate_sim()

    x = np.arange(0, TIME_STEPS+INPUT_SIZE-1)
    y = np.arange(INPUT_SIZE, TIME_STEPS+INPUT_SIZE+OUTPUT_SIZE-1)

    xs = [data[i].gy for i in x]
    ys = [ground[i].t_pitch for i in y]

    # Bandpass Filter
    processor = SignalProcessor(SAMPLE_RATE)
    x_filtered, _ = processor.Bandpass_Filter(xs, 3, 12, 5)

    for each in window(x_filtered, INPUT_SIZE):
        seq.append(each)
    for each in window(ys, OUTPUT_SIZE):
        gt.append(each)

    return [seq, gt, y]

def save_model(saver, sess, path, count):
    path += "{}".format(args.network)
    saver.save(sess, path, global_step=count)
    logging.debug("Model Saved with Count: " + count)

def load_model(saver, sess, path):
    saver.restore(sess, path)
    logging.debug("Model Loaded from Path: {}".format(path))

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)

if __name__ == "__main__":
    main()
