from __future__ import print_function

import os
import time
import json
import argparse
import densenet
import numpy as np
import keras.backend as K

from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = '/home/zixuan/neurocompann/Datasets/cats_vs_dogs'

def sample_latency_ANN(ann, batch_shape, repeat):
    samples = []

    # drop first run
    ann.predict(np.random.random(batch_shape), batch_size=batch_shape[0])

    for i in range(repeat):
        data_in = np.random.random(batch_shape)
        start_time = time.time()
        ann.predict(data_in, batch_size=batch_shape[0])
        samples.append(time.time() - start_time)
    per_frame_latency = np.array(samples) / batch_shape[0]
    avg_latency_per_frame = np.average(per_frame_latency)
    std_dev_per_frame = np.std(per_frame_latency)
    return(avg_latency_per_frame, std_dev_per_frame)

def run_gtsrb(batch_size,
                nb_epoch,
                depth,
                nb_dense_block,
                nb_filter,
                growth_rate,
                dropout_rate,
                learning_rate,
                weight_decay,
                logfile,
                plot_architecture):
    """ Run GTSRB experiments

    :param batch_size: int -- batch size
    :param nb_epoch: int -- number of training epochs
    :param depth: int -- network depth
    :param nb_dense_block: int -- number of dense blocks
    :param nb_filter: int -- initial number of conv filter
    :param growth_rate: int -- number of new filters added by conv layers
    :param dropout_rate: float -- dropout rate
    :param learning_rate: float -- learning rate
    :param weight_decay: float -- weight decay
    :param plot_architecture: bool -- whether to plot network architecture

    """

    ###################
    # Data processing #
    ###################
    tr_x = np.load(os.path.join(DATASET_DIR, 'rgb_train_in.npy'))
    tr_y = np.load(os.path.join(DATASET_DIR, 'rgb_train_out.npy'))
    te_x = np.load(os.path.join(DATASET_DIR, 'rgb_test_in.npy'))
    te_y = np.load(os.path.join(DATASET_DIR, 'rgb_test_out.npy'))
    # va_x = np.load(os.path.join(DATASET_DIR, 'rgb_valid_in.npy'))
    # va_y = np.load(os.path.join(DATASET_DIR, 'rgb_valid_out.npy'))
    X_train = tr_x
    Y_train = tr_y
    X_test = te_x
    Y_test = te_y

    nb_classes = Y_train.shape[1]
    img_dim = X_train.shape[1:]

    if K.image_data_format() == "channels_first":
        n_channels = X_train.shape[1]
    else:
        n_channels = X_train.shape[-1]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train * 1/255
    X_test = X_test * 1/255

    # Normalisation
    # X = np.vstack((X_train, X_test))
    # 2 cases depending on the image ordering
    # if K.image_data_format() == "channels_first":
    #     for i in range(n_channels):
    #         mean = np.mean(X[:, i, :, :])
    #         std = np.std(X[:, i, :, :])
    #         X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
    #         X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

    # elif K.image_data_format() == "channels_last":
    #     for i in range(n_channels):
    #         mean = np.mean(X[:, :, :, i])
    #         std = np.std(X[:, :, :, i])
    #         X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
    #         X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

    ###################
    # Construct model #
    ###################

    model = densenet.DenseNetImageNet121(input_shape=img_dim,
                                        dropout_rate = 0.2,
                                        weights=None,
                                        classes=nb_classes)
    # Model output
    model.summary()

    # Build optimizer
    # opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt = SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])

    # if plot_architecture:
    #     from keras.utils.visualize_util import plot
    #     plot(model, to_file='./figures/densenet_archi.png', show_shapes=True)

    ####################
    # Network profiling#
    ####################
    batch_shape = (batch_size, ) + img_dim
    repeat = 25

    model_latency, model_CI = sample_latency_ANN(model, batch_shape, repeat)
    print(model_latency)


    ####################
    # Network training #
    ####################

    print("Training")

    list_train_loss = []
    list_test_loss = []
    list_learning_rate = []

    datagen = ImageDataGenerator()

    for e in range(nb_epoch):

        if e == int(0.5 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

        if e == int(0.75 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))

        l_train_loss = []
        start = time.time()

        model.fit_generator(datagen.flow(X_train, Y_train, batch_size), epochs=1)

        test_logloss, test_acc = model.evaluate(X_test,
                                                Y_test,
                                                verbose=1,
                                                batch_size=64)
        list_test_loss.append([test_logloss, test_acc])
        list_learning_rate.append(float(K.get_value(model.optimizer.lr)))
        # to convert numpy array to json serializable
        print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

        d_log = {}
        d_log["batch_size"] = batch_size
        d_log["latency"] = model_latency
        d_log["nb_epoch"] = nb_epoch
        d_log["optimizer"] = opt.get_config()
        # d_log["train_loss"] = list_train_loss
        d_log["test_loss"] = list_test_loss
        d_log["learning_rate"] = list_learning_rate

        json_file = os.path.join('./log', logfile)
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run GTSRB experiment')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=30, type=int,
                        help='Number of epochs')
    parser.add_argument('--depth', type=int, default=7,
                        help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=1,
                        help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=16,
                        help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1E-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4,
                        help='L2 regularization on weights')
    parser.add_argument('--logfile', type=str, default='experiment_log_cifar10.json',
                        help='logfile name')
    parser.add_argument('--plot_architecture', type=bool, default=False,
                        help='Save a plot of the network architecture')

    args = parser.parse_args()

    print("Network configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(name, value)

    list_dir = ["./log", "./figures"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    run_gtsrb(args.batch_size,
                args.nb_epoch,
                args.depth,
                args.nb_dense_block,
                args.nb_filter,
                args.growth_rate,
                args.dropout_rate,
                args.learning_rate,
                args.weight_decay,
                args.logfile,
                args.plot_architecture)
