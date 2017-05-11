""" Inspired by example from
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
Uses the TensorFlow backend
The basic idea is to detect anomalies in a time-series.
"""
import pprint

import matplotlib.pyplot as plt
import numpy as np
import time
from random import randint
from pymongo import MongoClient
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Input, concatenate
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
from numpy import arange, sin, pi, random

import lstm

np.random.seed(1234)

# Global hyper-parameters
sequence_length = 1000
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs = 1
batch_size = 50


def label(x):
    result = np.zeros((1, 15))
    result[0, x] = 1.0
    return result


disease = {'Healthy control': label(0),
           'Valvular heart disease': label(1),
           'Dysrhythmia': label(2),
           'Myocardial infarction': label(3),
           'Heart failure (NYHA 2)': label(4),
           'Heart failure (NYHA 3)': label(5),
           'Heart failure (NYHA 4)': label(6),
           'Palpitation': label(7),
           'Cardiomyopathy': label(8),
           'Stable angina': label(9),
           'Hypertrophy': label(10),
           'Bundle branch block': label(11),
           'Unstable angina': label(12),
           'Myocarditis': label(13),
           'n/a': label(14)}


def dropin(X, y):
    """ The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
    :param X: Each row is a training sequence
    :param y: Tne target we train and will later predict
    :return: new augmented X, y
    """
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0, random_data_dup)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)


def generatorData(db, ifrom, ito):
    while 1:
        pData = db.physionetData.find()
        i = randint(0, 20000)
        index = randint(ifrom, ito)
        #print("\nerror:", index)
        record = pData.limit(-1).skip(index).next()
        inputdata1, mean1 = z_norm(record['ecgDataD1'][i:i + 10000])
        inputdata2, mean2 = z_norm(record['ecgDataD2'][i:i + 10000])
        inputdata3, mean3 = z_norm(record['ecgDataD3'][i:i + 10000])
        inputdata = [inputdata1, inputdata2, inputdata3]
        labeldata = disease[record['reason']]
        yield (inputdata, labeldata)


def getATestData(db, ifrom, ito):
    pData = db.physionetData.find()
    i = randint(0, 20000)
    index = randint(ifrom, ito)
    # print("\nerror:", index)
    record = pData.limit(-1).skip(index).next()
    inputdata1, mean1 = z_norm(record['ecgDataD1'][i:i + 10000])
    inputdata2, mean2 = z_norm(record['ecgDataD2'][i:i + 10000])
    inputdata3, mean3 = z_norm(record['ecgDataD3'][i:i + 10000])
    inputdata = [inputdata1, inputdata2, inputdata3]
    labeldata = disease[record['reason']]
    return inputdata, labeldata


def z_norm(result):
    result = np.array(result)
    result = np.reshape(result, (1, result.shape[0]))
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean


def build_model():
    layers = {'input': 10000, 'hidden1': 256, 'output': 15}
    inputD1 = Input(shape=(layers['input'],))
    inputD2 = Input(shape=(layers['input'],))
    inputD3 = Input(shape=(layers['input'],))
    d1 = Dense(layers['hidden1'], activation='sigmoid')(inputD1)
    d2 = Dense(layers['hidden1'], activation='sigmoid')(inputD2)
    d3 = Dense(layers['hidden1'], activation='sigmoid')(inputD3)
    d_all = concatenate([d1, d2, d3])
    output = Dense(layers['output'], activation='linear')(d_all)
    model = Model(inputs=[inputD1, inputD2, inputD3], outputs=output)

    # model = Sequential()
    # layers = {'input': 30000, 'hidden1': 256, 'output': 15}
    #
    # model.add(Dense(layers['hidden1'], input_shape=(layers['input'],)))
    # model.add(Dropout(0.2))
    # model.add(Dense(layers['output'], activation='relu'))
    # #model.add(Activation("tanh"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    print(model.summary())
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()

    if model is None:
        model = build_model()

    try:

        client = MongoClient()
        db = client.iotsystem
        count = db.physionetData.count()
        test_in, test_label = getATestData(db, count - 99, count - 1)
        print("Training...")
        model.fit_generator(generatorData(db, 0, count - 100), steps_per_epoch=200, epochs=5)

        metrics = model.evaluate_generator(generatorData(db, count - 99, count - 1), steps=100)
        print(model.metrics_names)
        print(metrics)
        print("target:    ", test_label)
        print("predicted: ", model.predict(test_in))


    except KeyboardInterrupt:
        print("prediction exception")
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, 0

    try:
        cc = 1
    except Exception as e:
        print("plotting exception")
        print(str(e))
    print('Training duration (s) : ', time.time() - global_start_time)

    return model


run_network()
