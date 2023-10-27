#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[ ]:
import codecs

from numpy import interp
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool1D, GRU, LSTM
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential
import tensorflow as tf
import csv
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,  roc_curve
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
import hickle as hkl
import copy
from scipy.spatial import distance_matrix
import sys
import random
import math
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from itertools import cycle
from tensorflow.keras.utils import to_categorical

# In[ ]:


# which GPU to use
# "-1,0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])


# DNN,CNN，GRU,LSTM
modelType = "LSTM"


algorithm = "PSPFL"

# REALWORLD_CLIENT,sisfall
dataSetName = 'REALWORLD_CLIENT'

# BALANCED, UNBALANCED
dataConfig = "BALANCED"

# ADAM, SGD
optimizer = "SGD"

# Have model evaluate on the Global testset
ClientAllTest = True

# Neuron distance measurement 
euclid = True

# Asynchronous client test
asyncTest = False
ratio=0.5
rate = 1
# if 0, uses 33% as starting pool
startingTrainPool = 0

clientDeleteCount = 3
clientAddCount = 5

# only needed to set if clientAddCount = clientDeleteCount, otherwise it follows a small algorithm to calculate automatically
asyncInterval = 5

# Generate results in seperate graph
seperateGraph = False

# Save the client models a .h5 file
savedClientModel = 0

# Show training verbose: 0,1
showTrainVerbose = 0

# input window size 
segment_size = 128

# input channel count
num_input_channels = 6

# 屏蔽比率
level = 0

# client learning rate
learningRate = 0.01

# model drop out rate
dropout_rate = 0.5

# local epoch
localEpoch = 4

# communication round
communicationRound = 100

# Seed for data partioning and TF training
randomSeed = 1

# In[ ]:


# specifying activities and where the results will be stored 


if (dataSetName == 'REALWORLD_CLIENT'):
    ACTIVITY_LABEL = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
else:
    ACTIVITY_LABEL = ['D01', 'D03', 'D05', 'D07', 'D02', 'D04', 'D09', 'D08']
    num_input_channels = 9
activityCount = len(ACTIVITY_LABEL)

architectureType = str(algorithm) + '_' + 'LR_' + str(localEpoch) + 'LE_' + str(communicationRound) + 'CR_' + str(
    modelType)+"_Fed_wei_personalized_former_nodelete_f_test_" + str(level)+'_'+str(rate)+'_'+str(ratio)+'one_age_male'

mainDir = ''
filepath = mainDir + 'savedModels/' + architectureType + '/' + dataSetName + '/'
os.makedirs(filepath, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
if (dataSetName == 'UCI'):
    clientCount = 10
elif (dataSetName == 'REALWORLD_CLIENT'):
    clientCount = 30
else:
    clientCount = 24
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
np.random.seed(randomSeed)
tf.random.set_seed(randomSeed)

# In[ ]:


# Initializing data variables

clientDataTrain = []
clientLabelTrain = []
clientDataVali = []
clientLabelVali = []
clientDataTest = []
clientLabelTest = []

centralTrainData = []
centralTrainLabel = []

centralTestData = []
centralTestLabel = []
# 改
finalTestData =[]
finalTestLabel = []
f_clientDataTest = []
f_clientLabelTest = []

# In[ ]:


# loading datasets


if (dataSetName == 'REALWORLD_CLIENT'):
    clientData = []
    clientLabel = []

    dataSetName = 'REALWORLD_CLIENT'
    for i in range(0, 30):
        accX = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/AccX' + dataSetName + '.hkl')
        accY = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/AccY' + dataSetName + '.hkl')
        accZ = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/AccZ' + dataSetName + '.hkl')
        gyroX = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/GyroX' + dataSetName + '.hkl')
        gyroY = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/GyroY' + dataSetName + '.hkl')
        gyroZ = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/GyroZ' + dataSetName + '.hkl')
        label = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/Label' + dataSetName + '.hkl')
        clientData.append(np.dstack((accX, accY, accZ, gyroX, gyroY, gyroZ)))
        clientLabel.append(label)
    for i in range(0,30):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomSeed)
        skf.get_n_splits(clientData[i], clientLabel[i])
        partionData = list()
        partionLabel = list()
        for train_index, test_index in skf.split(clientData[i], clientLabel[i]):
            partionData.append(clientData[i][test_index])
            partionLabel.append(clientLabel[i][test_index])
        clientData[i] = partionData[0]
        clientLabel[i] = partionLabel[0]
    if (dataConfig == "BALANCED"):
        for i in range(0, 30):
            skf = StratifiedKFold(n_splits=11, shuffle=True, random_state=randomSeed)
            skf.get_n_splits(clientData[i], clientLabel[i])
            partitionedData = list()
            partitionedLabel = list()
            for train_index, test_index in skf.split(clientData[i], clientLabel[i]):
                partitionedData.append(clientData[i][test_index])
                partitionedLabel.append(clientLabel[i][test_index])
            clientDataTrain.append((np.vstack((partitionedData[:8]))))
            clientLabelTrain.append((np.hstack((partitionedLabel[:8]))))
            clientDataVali.append((partitionedData[8]))
            clientLabelVali.append((partitionedLabel[8]))
            clientDataTest.append((partitionedData[9]))
            clientLabelTest.append((partitionedLabel[9]))
            f_clientDataTest.append(partitionedData[10])
            f_clientLabelTest.append(partitionedLabel[10])
    else:
        for i in range(0, 30):
            kf = KFold(n_splits=5, shuffle=True, random_state=randomSeed)
            kf.get_n_splits(clientData[i])
            partitionedData = list()
            partitionedLabel = list()
            for train_index, test_index in kf.split(clientData[i]):
                partitionedData.append(clientData[i][test_index])
                partitionedLabel.append(clientLabel[i][test_index])
            clientDataTrain.append((np.vstack((partitionedData[:4]))))
            clientLabelTrain.append((np.hstack((partitionedLabel[:4]))))
            clientDataTest.append((partitionedData[4]))
            clientLabelTest.append((partitionedLabel[4]))
    centralTrainData = (np.vstack((clientDataTrain)))
    centralTrainLabel = (np.hstack((clientLabelTrain)))
    centralValiData = (np.vstack(clientDataVali))
    centralValiLabel = (np.hstack(clientLabelVali))
    centralTestData = (np.vstack((clientDataTest)))
    centralTestLabel = (np.hstack((clientLabelTest)))
    if modelType == "GRU" or modelType == "LSTM":
        centralTrainLabel = to_categorical(centralTrainLabel, num_classes=8)
        centralValiLabel = to_categorical(centralValiLabel, num_classes=8)
        centralTestLabel = to_categorical(centralTestLabel, num_classes=8)
        for i in range(clientCount):
            clientLabelTrain[i] = to_categorical(clientLabelTrain[i], num_classes=8)
            clientLabelVali[i] = to_categorical(clientLabelVali[i], num_classes=8)
            clientLabelTest[i] = to_categorical(clientLabelTest[i], num_classes=8)
            # 改
            f_clientLabelTest[i] = to_categorical(f_clientLabelTest[i], num_classes=8)
else:
    clientData = []
    clientLabel = []

    dataSetName = 'sisifall'
    for i in range(0, 24):
        accX = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/AccX' + dataSetName + '.hkl')
        accY = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/AccY' + dataSetName + '.hkl')
        accZ = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/AccZ' + dataSetName + '.hkl')
        acc1X = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/Acc1X' + dataSetName + '.hkl')
        acc1Y = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/Acc1Y' + dataSetName + '.hkl')
        acc1Z = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/Acc1Z' + dataSetName + '.hkl')
        gyroX = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/GyroX' + dataSetName + '.hkl')
        gyroY = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/GyroY' + dataSetName + '.hkl')
        gyroZ = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/GyroZ' + dataSetName + '.hkl')
        label = hkl.load('datasetStandardized/' + dataSetName + '/' + str(i) + '/Label' + dataSetName + '.hkl')
        clientData.append(np.dstack((accX, accY, accZ, acc1X, acc1Y, acc1Z,gyroX, gyroY, gyroZ)))
        clientLabel.append(label)
    for i in range(0, 24):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomSeed)
        skf.get_n_splits(clientData[i], clientLabel[i])
        partionData = list()
        partionLabel = list()
        for train_index, test_index in skf.split(clientData[i], clientLabel[i]):
            partionData.append(clientData[i][test_index])
            partionLabel.append(clientLabel[i][test_index])
        clientData[i] = partionData[0]
        clientLabel[i] = partionLabel[0]
    if (dataConfig == "BALANCED"):
        for i in range(0, 24):
            skf = StratifiedKFold(n_splits=11, shuffle=True, random_state=randomSeed)
            skf.get_n_splits(clientData[i], clientLabel[i])
            partitionedData = list()
            partitionedLabel = list()
            for train_index, test_index in skf.split(clientData[i], clientLabel[i]):
                partitionedData.append(clientData[i][test_index])
                partitionedLabel.append(clientLabel[i][test_index])
            clientDataTrain.append((np.vstack((partitionedData[:8]))))
            clientLabelTrain.append((np.hstack((partitionedLabel[:8]))))
            clientDataVali.append((partitionedData[8]))
            clientLabelVali.append((partitionedLabel[8]))
            clientDataTest.append((partitionedData[9]))
            clientLabelTest.append((partitionedLabel[9]))
            f_clientDataTest.append(partitionedData[10])
            f_clientLabelTest.append(partitionedLabel[10])
    else:
        for i in range(0, 24):
            kf = KFold(n_splits=5, shuffle=True, random_state=randomSeed)
            kf.get_n_splits(clientData[i])
            partitionedData = list()
            partitionedLabel = list()
            for train_index, test_index in kf.split(clientData[i]):
                partitionedData.append(clientData[i][test_index])
                partitionedLabel.append(clientLabel[i][test_index])
            clientDataTrain.append((np.vstack((partitionedData[:4]))))
            clientLabelTrain.append((np.hstack((partitionedLabel[:4]))))
            clientDataTest.append((partitionedData[4]))
            clientLabelTest.append((partitionedLabel[4]))
    centralTrainData = (np.vstack((clientDataTrain)))
    centralTrainLabel = (np.hstack((clientLabelTrain)))
    centralValiData = (np.vstack(clientDataVali))
    centralValiLabel = (np.hstack(clientLabelVali))
    centralTestData = (np.vstack((clientDataTest)))
    centralTestLabel = (np.hstack((clientLabelTest)))
    if modelType == "GRU" or modelType == "LSTM":
        centralTrainLabel = to_categorical(centralTrainLabel, num_classes=8)
        centralValiLabel = to_categorical(centralValiLabel, num_classes=8)
        centralTestLabel = to_categorical(centralTestLabel, num_classes=8)
        for i in range(clientCount):
            clientLabelTrain[i] = to_categorical(clientLabelTrain[i], num_classes=8)
            clientLabelVali[i] = to_categorical(clientLabelVali[i], num_classes=8)
            clientLabelTest[i] = to_categorical(clientLabelTest[i], num_classes=8)
            # 改
            f_clientLabelTest[i] = to_categorical(f_clientLabelTest[i], num_classes=8)
# In[ ]:




if (modelType == 'LSTM'):
    def create_keras_model():
        return Sequential([
            LSTM(120, input_shape=(segment_size, num_input_channels)),
            Dropout(dropout_rate),
            Dense(120, activation='relu'),
            Dense(activityCount, activation='softmax')
        ])


    serverModel = Sequential()
    serverModel.add(LSTM(120, input_shape=(segment_size, num_input_channels)))
    serverModel.add(Dropout(dropout_rate))
    serverModel.add(Dense(120, activation='relu'))
    serverModel.add(Dense(activityCount, activation='softmax'))
# In[ ]:


# initializing DNN model
if (modelType == "DNN"):
    def create_keras_model():
        return Sequential([
            Flatten(input_shape=(segment_size, num_input_channels), name='flatten'),
            Dense(units=400, activation='relu', name='base'),
            Dropout(dropout_rate),
            Dense(units=100, activation='relu', name='personalized'),
            Dropout(dropout_rate),
            Dense(activityCount, activation='softmax', name='clientSoft')
        ])


    # initialize server
    serverModel = Sequential()
    serverModel.add(Flatten(input_shape=(segment_size, num_input_channels), name='flatten'))
    serverModel.add(Dense(400, activation='relu', name='base'))
    serverModel.add(Dropout(dropout_rate))
    serverModel.add(Dense(100, activation='relu', name='serverP'))
    serverModel.add(Dropout(dropout_rate))
    serverModel.add(Dense(activityCount, activation='softmax', name='serverSoft'))

# In[ ]:


# compiling the server model
if (optimizer == "SGD"):
    if(modelType=='CNN' or modelType=='DNN'):
        serverModel.compile(optimizer=SGD(learning_rate=learningRate), loss='sparse_categorical_crossentropy', metrics=['acc'])
    else:
        serverModel.compile(optimizer=SGD(learning_rate=learningRate), loss='categorical_crossentropy',
                            metrics=['acc'])
else:
    serverModel.compile(optimizer=Adam(learning_rate=learningRate), loss='sparse_categorical_crossentropy',metrics=['acc'])

if not (os.path.exists(filepath + 'serverWeights.h5')):
    print('already not have')
    serverModel.save_weights(filepath + 'serverWeights.h5')
    weights = serverModel.get_weights()

# In[ ]:


# initializing client model
local_nets = {}
local_histories = {}

for i in range(0, clientCount):
    local_nets[i] = create_keras_model()

# In[ ]:


# Initialization of metrics during training
# client models test againts own test-set
trainLossHistory = []
valiLossHistory = []
trainAccHistory = []
testLossHistory = []
testAccHistory = []
if(os.path.exists(filepath+'trainingStats/trainLossHistory.hkl')):

    trainLossHistory=hkl.load(filepath+'trainingStats/trainLossHistory.hkl').tolist()
    valiLossHistory=hkl.load(filepath+'trainingStats/trainLossHistory.hkl').tolist()
    print(len(trainLossHistory))
    if (len(trainLossHistory) != 100 - communicationRound):
        trainLossHistory = trainLossHistory[:100 - communicationRound]
        valiLossHistory=valiLossHistory[:100 - communicationRound]
stdTrainLossHistory = []
stdValiLossHistory = []
stdTrainAccHistory = []
stdTestLossHistory = []
stdTestAccHistory = []

# client models test againts all test-set

clientTrainLossHistory = []
clientTrainAccHistory = []
clientTestLossHistory = []
clientTestAccHistory = []

clientStdTrainLossHistory = []
clientStdTrainAccHistory = []
clientStdTestLossHistory = []
clientStdTestAccHistory = []

# server test againts all test-set

serverTrainLossHistory = []
serverTrainAccHistory = []
serverTestLossHistory = []
serverTestAccHistory = []

meanHistoryDist = []
stdHistoryDist = []

meanRoundLayerHistory = []
stdRoundLayerHistory = []

meanRoundGeneralLayerHistory = []
stdRoundGeneralLayerHistory = []

bestModelRound = 0
currentAccuracy = 0.0
serverCurrentAccuracy = 0.0
serverbestModelRound = 0
bestServerModel = None
bestServerModelWeights = None
best_local_nets = {}

best_local_acc = []
for i in range(clientCount):
    best_local_acc.append(0.0)
    if (os.path.exists(filepath + "bestlocal_nets[" + str(i) + "].h5")):
        if (optimizer == "SGD"):
            if (modelType == 'CNN' or modelType == 'DNN'):
                local_nets[i].compile(optimizer=SGD(learning_rate=learningRate), loss='sparse_categorical_crossentropy',
                                      metrics=['acc'])
            else:
                local_nets[i].compile(optimizer=SGD(learning_rate=learningRate), loss='categorical_crossentropy',
                                      metrics=['acc'])

        else:
            local_nets[i].compile(optimizer=Adam(learning_rate=learningRate), loss='sparse_categorical_crossentropy',
                                  metrics=['acc'])
        local_nets[i].load_weights(filepath + "bestlocal_nets[" + str(i) + "].h5")
        f_ModelMetrics = local_nets[i].evaluate(f_clientDataTest[i], f_clientLabelTest[i], verbose=showTrainVerbose)
        best_local_acc[i] = f_ModelMetrics[1]
        best_local_nets[i] = copy.copy(local_nets[i])


best_local_weights = {}

stage = 1
clientParticipant = clientCount

# In[ ]:


# Generates an array to represent model type per layer
layerType = []
for idx in range(len(serverModel.layers)):
    temp = serverModel.get_layer(index=idx).__class__.__name__
    if ("Conv" in temp):
        layerType.append(0)
    elif ("Dense" in temp):
        layerType.append(1)




# calculating dataset size weight per client
local_coeffs = {}

# In[ ]:


# calculating activities weight for weighted training per client
if(modelType=='CNN' or modelType=='DNN' ):
    local_class_weights = {}
    for i in range(0, clientCount):
        temp_weights = class_weight.compute_class_weight('balanced',
                                                         np.unique(clientLabelTrain[i]),
                                                         clientLabelTrain[i].ravel())
        local_class_weights[i] = {j: temp_weights[j] for j in range(len(temp_weights))}
else:
    local_class_weights = [None for i in range(clientCount)]
    for i in range(clientCount):
        temp_weights = class_weight.compute_class_weight('balanced'
                                                         , classes=np.unique(clientLabelTrain[i].argmax(axis=-1))
                                                         , y=clientLabelTrain[i].argmax(axis=-1)).tolist()
        local_class_weights[i] = {j: temp_weights[j] for j in range(len(temp_weights))}
# In[ ]:


# initialization for asynchronous client training, client selection
roundEnd = []
if (asyncTest):
    trainPool = []
    idlePool = []

    if (startingTrainPool == 0):
        initialClientCount = int(clientCount * 0.34)
        if (initialClientCount == 1):
            initialClientCount = 2
    else:
        initialClientCount = startingTrainPool

    trainPool = list(range(initialClientCount))
    idlePool = list(range(initialClientCount, clientCount))

    if (clientDeleteCount != clientAddCount):
        stages = math.ceil((clientCount - len(trainPool)) / (clientAddCount - clientDeleteCount))
        intervals = int(communicationRound / (stages * 2))
    else:
        intervals = asyncInterval
        stages = int(communicationRound / intervals)
    for clientChangeRound in range(1, stages + 1):
        roundEnd.append(intervals * clientChangeRound)
else:
    trainPool = range(clientCount)

# In[ ]:


# initialization of client distance
clientEuclidDistMean = {}
clientEuclidDistStd = {}
for i in range(clientCount):
    clientEuclidDistMean[i] = np.zeros(communicationRound)
    clientEuclidDistStd[i] = np.zeros(communicationRound)
# Calculate similarity between clients
subjects = []
if (dataSetName == 'REALWORLD_CLIENT'):
    file = 'realworld1.csv'
else:
    file = 'sisfall.csv'
with codecs.open(file) as f:
    for line in f.readlines():
        subject = []
        line = line.replace('\ufeff', '').replace('\n', '')
        line = line.split(',')
        for i in range(len(line)):
            subject.append(float(line[i]))
        subjects.append(subject)
subjects = np.asarray(subjects, dtype=np.float)
new_data = np.zeros(shape=(len(subjects), len(subjects[1])))
for i in range(len(subjects)):
    for j in range(len(subjects[i])):
        data_min = min(subjects[:, j])
        data_max = max(subjects[:, j])
        new_data[i][j] = (subjects[i][j] - data_min) / (data_max - data_min)
cos_sim = np.zeros(shape=(len(subjects), len(subjects)))
for i in range(len(subjects)):
    for j in range(len(subjects)):
        cos_sim[i][j] = cosine_similarity(new_data[i].reshape(1, -1), new_data[j].reshape(1, -1))

# In[ ]:

global_acc = 0
# Federated learning training
for roundNum in range(0, communicationRound):
    total = 0
    flag = []
    truelabel = 0
    sim = cos_sim.copy()
    # start_time = time.time()
    trainAcc = []
    trainLoss = []
    valiLoss = []
    # 改
    testAcc = []
    # former_acc = []
    former_acc1 = []
    # former_acc2 = []
    # former_acc3 = []
    for i in range(clientCount):
        testAcc.append(0.0)
        # former_acc.append(0.0)
        former_acc1.append(0.0)
        # former_acc2.append(0.0)
        # former_acc3.append(0.0)
    testLoss = []

    clientTrainAcc = []
    clientTrainLoss = []

    clientTestAcc = []
    clientTestLoss = []

    local_weights = {}
    # Similarity model parameters
    p_local_weights = {}
    # Model parameters calculated according to proportion of data volume
    if (asyncTest):
        if (roundNum in roundEnd):
            for i in range(clientDeleteCount):
                if (len(trainPool) != 0):
                    selection = random.choice(list(enumerate(trainPool)))
                    del trainPool[selection[0]]
                    idlePool.append(selection[1])
            for i in range(clientAddCount):
                if (len(idlePool) != 0):
                    selection = random.choice(list(enumerate(idlePool)))
                    del idlePool[selection[0]]
                    trainPool.append(selection[1])

        participantDataInstance = []
        for index, i in enumerate(trainPool):
            participantDataInstance.append(clientLabelTrain[i])
        participantDataInstance = (np.hstack((participantDataInstance)))
        local_coeffs = {}
        for index, i in enumerate(trainPool):
            local_coeffs[i] = np.float32(len(clientLabelTrain[i])) / np.float32(len(participantDataInstance))
    seq = [i for i in range(clientCount)]
    selected = int(clientCount * (1. - level) + 0.5)
    print('SLECTED',selected)
    a = random.sample(seq, selected)
    a.sort()
    for i in range(clientCount):
        if (i in a):
            flag.append(1)
            truelabel += 1
        else:
            flag.append(0)
    print(flag)

    min_array = []
    min_flag = -1
    for index, i in enumerate(trainPool):
        if (flag[i] == 1):

            print("Status: Round #" + str(roundNum) + " Client #" + str(i))


            if (os.path.exists(filepath + "local_nets[" + str(i) + "].h5")):
                local_nets[i].load_weights(filepath + "local_nets[" + str(i) + "].h5")
            else:
                local_nets[i].load_weights(filepath + 'serverWeights.h5', by_name=False)
            if (optimizer == "SGD"):
                if (modelType == 'CNN' or modelType == 'DNN'):
                    local_nets[i].compile(optimizer=SGD(learning_rate=learningRate),loss='sparse_categorical_crossentropy', metrics=['acc'])
                else:
                    local_nets[i].compile(optimizer=SGD(learning_rate=learningRate), loss='categorical_crossentropy',metrics=['acc'])

            else:
                local_nets[i].compile(optimizer=Adam(learning_rate=learningRate),loss='sparse_categorical_crossentropy',
                                          metrics=['acc'])

            local_histories[i] = local_nets[i].fit(clientDataTrain[i], clientLabelTrain[i],
                                                   epochs=localEpoch,
                                                   verbose=showTrainVerbose,
                                                   validation_data=(clientDataVali[i], clientLabelVali[i]))
            local_weights[i] = local_nets[i].get_weights()
            p_local_weights[i] = local_nets[i].get_weights()
            trainAcc.append(local_histories[i].history['acc'])
            trainLoss.append(local_histories[i].history['loss'])
            valiLoss.append(local_histories[i].history['val_loss'])

            testModelMetrics = local_nets[i].evaluate(centralValiData,centralValiLabel, verbose=showTrainVerbose)


            f_ModelMetrics = local_nets[i].evaluate(f_clientDataTest[i], f_clientLabelTest[i], verbose=showTrainVerbose)
            former_acc1[i] = f_ModelMetrics[1]


            testLoss.append(testModelMetrics[0])

            total = total + len(clientLabelTrain[i])


    print("f_ModelMetrics", former_acc1)

    for i in range(clientCount):
        for j in range(clientCount):

            if flag[j] == 0 or j == min_flag or flag[i] == 0:
                sim[i][j] = 0
                if i == j and i == min_flag:
                    sim[i][j] = cos_sim[i][j]
    arr = np.sum(sim, axis=1)

    for i in range(clientCount):
        for j in range(clientCount):
            if flag[j] == 1 and flag[i] == 1 and i != j:
                sim[i][j] = sim[i][j] / (arr[i] - sim[i][i])
        for j in range(clientCount):
            if flag[i] == 1 and flag[j] == 1 and i != min_flag and j != min_flag:
                for z in range(0, len(local_weights[j])):
                    if (i != j):
                        p_local_weights[i][z] += local_weights[j][z] * sim[i][j] * rate * ratio
                    if (i == j):
                        p_local_weights[i][z] -= local_weights[i][z]
                        p_local_weights[i][z] += local_weights[j][z] * sim[i][j] * rate * (1 - ratio)
            if (i == min_flag and flag[j] == 1):
                for z in range(0, len(local_weights[j])):
                    p_local_weights[i][z] += local_weights[j][z] * sim[i][j]
                    if (i == j):
                        p_local_weights[i][z] -= local_weights[i][z]

    for index, i in enumerate(trainPool):
        if (i == min_flag):
            total = total - len(clientLabelTrain[i])
    for index, i in enumerate(trainPool):
        if (flag[i] == 1):
            if (i != min_flag):
                local_coeffs = (np.float32(len(clientLabelTrain[i])) / np.float32(total))
                for j in range(0, len(local_weights[i])):
                    local_weights[i][j] = local_weights[i][j] * local_coeffs * (1 - rate)


    trainAccHistory.append(np.mean(trainAcc))
    stdTrainAccHistory.append(np.std(trainAcc))
    if(type(trainLossHistory)!=list):
        trainLossHistory=trainLossHistory.tolist()
        valiLossHistory=valiLossHistory.tolist()
        stdTrainLossHistory=stdTrainLossHistory.tolist()
        stdValiLossHistory =stdValiLossHistory.tolist()
    trainLossHistory.append(np.mean(trainLoss))

    valiLossHistory.append(np.mean(valiLoss))


    stdTrainLossHistory.append(np.std(trainLoss))
    stdValiLossHistory.append(np.std(valiLoss))
    stdTrainLossHistory = np.asarray(stdTrainLossHistory)
    stdValiLossHistory = np.asarray(stdValiLossHistory)
    trainLossHistory = np.asarray(trainLossHistory)
    valiLossHistory = np.asarray(valiLossHistory)
    if(not os.path.exists(filepath + "trainingStats/trainLossHistory.hkl")):
        os.makedirs(filepath + 'trainingStats', exist_ok=True)
    hkl.dump(trainLossHistory, filepath + "trainingStats/trainLossHistory.hkl")
    hkl.dump(valiLossHistory, filepath + "trainingStats/valiLossHistory.hkl")
    hkl.dump(stdTrainLossHistory, filepath + "trainingStats/stdTrainLossHistory.hkl")
    hkl.dump(stdValiLossHistory, filepath + "trainingStats/stdValiLossHistory.hkl")


    meanTestAcc = np.mean(testAcc)

    testAccHistory.append(meanTestAcc)
    stdTestAccHistory.append(np.std(testAcc))
    testLossHistory.append(np.mean(testLoss))
    stdTestLossHistory.append(np.std(testLoss))


    for index, net in enumerate(local_nets):
        if (flag[index] == 1):
            if (former_acc1[index] > best_local_acc[index]):
                best_local_nets[index] = copy.copy(local_nets[index])
                local_nets[index].save_weights(filepath + "bestlocal_nets[" + str(index) + "].h5")
                best_local_acc[index] = former_acc1[index]



    # return weights to server and sum all the model weights 
    if roundNum != communicationRound - 1:
        weights = []
        if (truelabel != 1):
            for i in local_weights:
                if (i != min_flag):
                    if (flag[i] == 1):
                        # print(1)
                        weights.append(local_weights[i])

                else:
                    pass
        else:
            for i in local_weights:
                if (flag[i] == 1):
                    # print(1)
                    weights.append(local_weights[i])
        new_weights = list()
        for weights_list_tuple in zip(*weights):
            new_weights.append(np.asarray(
                [np.array(weights_).sum(axis=0) \
                 for weights_ in zip(*weights_list_tuple)]))

        for i in range(clientCount):
            if (flag[i] == 1 and i != min_flag):
                for j in range(len(local_weights[i])):
                    p_local_weights[i][j] += new_weights[j]
                local_nets[i].set_weights(np.asarray(p_local_weights[i]))
                local_nets[i].save_weights(filepath + "local_nets[" + str(i) + "].h5")

            if(i==min_flag):
                local_nets[i].set_weights(np.asarray(p_local_weights[i]))
                valiModelMetrics = local_nets[i].evaluate(f_clientDataTest[i], f_clientLabelTest[i], verbose=showTrainVerbose)
                print(valiModelMetrics[1])
                if (valiModelMetrics[1] > former_acc1[i]):

    weights_server = []
    ser_local_weight = p_local_weights.copy()
    for index, i in enumerate(trainPool):
        if (flag[i] == 1):
            if (i != min_flag):
                local_coeffs = (np.float32(len(clientLabelTrain[i])) / np.float32(total))
                for j in range(0, len(ser_local_weight[i])):
                    ser_local_weight[i][j] = ser_local_weight[i][j] * local_coeffs
    if (truelabel != 1):

        for i in ser_local_weight:
            if (i != min_flag):
                if (flag[i] == 1):
                    # print(1)
                    weights_server.append(ser_local_weight[i])
            else:
                pass
    else:
        for i in ser_local_weight:
            if (flag[i] == 1):
                # print(1)
                weights_server.append(ser_local_weight[i])
    new_weights_server = list()
    for weights_list_tuple in zip(*weights_server):
        new_weights_server.append(np.asarray(
            [np.array(weights_).sum(axis=0) \
             for weights_ in zip(*weights_list_tuple)]))
    serverModel.set_weights(np.asarray(new_weights_server))
    serverModel.save_weights(filepath+'serverWeights.h5')


for index in range(len(roundEnd)):
    roundEnd[index] += 1

# In[ ]:


# convert datatypes to a np formats
# std of all clients
stdTrainLossHistory = np.asarray(stdTrainLossHistory)
stdValiLossHistory = np.asarray(stdValiLossHistory)
stdTrainAccHistory = np.asarray(stdTrainAccHistory)
stdTestLossHistory = np.asarray(stdTestLossHistory)
stdTestAccHistory = np.asarray(stdTestAccHistory)

clientStdTrainLossHistory = np.asarray(clientStdTrainLossHistory)
clientStdTrainAccHistory = np.asarray(clientStdTrainAccHistory)
clientStdTestLossHistory = np.asarray(clientStdTestLossHistory)
clientStdTestAccHistory = np.asarray(clientStdTestAccHistory)

if (euclid):
    meanHistoryDist = np.asarray(meanHistoryDist).T
    stdHistoryDist = np.asarray(stdHistoryDist).T
    meanRoundLayerHistory = np.asarray(meanRoundLayerHistory).T
    stdRoundLayerHistory = np.asarray(stdRoundLayerHistory).T
    meanRoundGeneralLayerHistory = np.asarray(meanRoundGeneralLayerHistory)
    stdRoundGeneralLayerHistory = np.asarray(stdRoundGeneralLayerHistory)
# mean
trainLossHistory = np.asarray(trainLossHistory)
valiLossHistory = np.asarray(valiLossHistory)
trainAccHistory = np.asarray(trainAccHistory)
testLossHistory = np.asarray(testLossHistory)
testAccHistory = np.asarray(testAccHistory)

clientTrainLossHistory = np.asarray(clientTrainLossHistory)
clientTrainAccHistory = np.asarray(clientTrainAccHistory)
clientTestLossHistory = np.asarray(clientTestLossHistory)
clientTestAccHistory = np.asarray(clientTestAccHistory)

if (algorithm != 'FEDPER'):
    serverTrainLossHistory = np.asarray(serverTrainLossHistory)
    serverTrainAccHistory = np.asarray(serverTrainAccHistory)
    serverTestLossHistory = np.asarray(serverTestLossHistory)
    serverTestAccHistory = np.asarray(serverTestAccHistory)

# In[ ]:


# Saving the training statistics and results
os.makedirs(filepath + 'trainingStats', exist_ok=True)

hkl.dump(trainLossHistory, filepath + "trainingStats/trainLossHistory.hkl")
hkl.dump(valiLossHistory, filepath + "trainingStats/valiLossHistory.hkl")
hkl.dump(trainAccHistory, filepath + "trainingStats/trainAccHistory.hkl")
hkl.dump(stdTrainLossHistory, filepath + "trainingStats/stdTrainLossHistory.hkl")
hkl.dump(stdValiLossHistory, filepath + "trainingStats/stdValiLossHistory.hkl")
hkl.dump(stdTrainAccHistory, filepath + "trainingStats/stdTrainAccHistory.hkl")

hkl.dump(testLossHistory, filepath + "trainingStats/testLossHistory.hkl")
hkl.dump(testAccHistory, filepath + "trainingStats/testAccHistory.hkl")
hkl.dump(stdTestLossHistory, filepath + "trainingStats/stdTestLossHistory.hkl")
hkl.dump(stdTestAccHistory, filepath + "trainingStats/stdTestAccHistory.hkl")

if (euclid):
    hkl.dump(meanHistoryDist.tolist(), filepath + "trainingStats/meanHistoryDist.hkl")
    hkl.dump(stdHistoryDist.tolist(), filepath + "trainingStats/stdHistoryDist.hkl")
    hkl.dump(meanRoundLayerHistory, filepath + "trainingStats/meanRoundLayerHistory.hkl")
    hkl.dump(stdRoundLayerHistory, filepath + "trainingStats/stdRoundLayerHistory.hkl")
    hkl.dump(meanRoundGeneralLayerHistory, filepath + "trainingStats/meanRoundGeneralLayerHistory.hkl")
    hkl.dump(stdRoundGeneralLayerHistory, filepath + "trainingStats/stdRoundGeneralLayerHistory.hkl")

if (ClientAllTest == True):
    hkl.dump(clientStdTrainLossHistory, filepath + "trainingStats/clientStdTrainLossHistory.hkl")
    hkl.dump(clientStdTrainAccHistory, filepath + "trainingStats/clientStdTrainAccHistory.hkl")
    hkl.dump(clientStdTestLossHistory, filepath + "trainingStats/clientStdTestLossHistory.hkl")
    hkl.dump(clientStdTestAccHistory, filepath + "trainingStats/clientStdTestAccHistory.hkl")

    hkl.dump(clientTrainLossHistory, filepath + "trainingStats/clientTrainLossHistory.hkl")
    hkl.dump(clientTrainAccHistory, filepath + "trainingStats/clientTrainAccHistory.hkl")
    hkl.dump(clientTestLossHistory, filepath + "trainingStats/clientTestLossHistory.hkl")
    hkl.dump(clientTestAccHistory, filepath + "trainingStats/clientTestAccHistory.hkl")

if (algorithm != 'FEDPER'):
    hkl.dump(serverTrainLossHistory, filepath + "trainingStats/serverTrainLossHistory.hkl")
    hkl.dump(serverTrainAccHistory, filepath + "trainingStats/serverTrainAccHistory.hkl")
    hkl.dump(serverTestLossHistory, filepath + "trainingStats/serverTestLossHistory.hkl")
    hkl.dump(serverTestAccHistory, filepath + "trainingStats/serverTestAccHistory.hkl")


# In[ ]:


# generate line chart function
def saveGraph(title="", accuracyOrLoss="Accuracy", asyTest=False, legendLoc='lower right'):
    if (asyTest):
        for stage in range(len(roundEnd)):
            plt.axvline(roundEnd[stage], 0, 1, color="blue")
    plt.title(title)
    plt.ylabel(accuracyOrLoss)
    plt.xlabel('Communication Round')
    plt.legend(loc=legendLoc)
    plt.savefig(filepath + title.replace(" ", "") + '.png', dpi=100)
    plt.clf()




# cohenD effect size normalize function
def cohenDNormalize(mean1, mean2, std1, std2):
    numerator = (mean1 - mean2)
    denominater = np.sqrt(((std1 ** 2) + (std2 ** 2) / 2))
    cohenDs = numerator / denominater
    meanNormalized = mean1 * cohenDs
    stdNormalized = std1 * cohenDs
    return meanNormalized, stdNormalized


# In[ ]:




# Rounding number function 
def roundNumber(toRoundNb):
    return round(np.mean(toRoundNb), 4)


# In[ ]:


# Generating personalized accuracy
indiAccTest = []
indiMacroPrecision=[]
indiMacroRecall=[]

indiWeightedTest = []
indiMicroTest = []
indiMacroTest = []

os.makedirs(filepath + 'models/', exist_ok=True)
for i in range(len(best_local_nets)):
    if(modelType=='CNN' or modelType=='DNN'):
        best_local_nets[i].compile(optimizer=SGD(learning_rate=learningRate), loss='sparse_categorical_crossentropy',
                               metrics=['acc'])
    else:
        best_local_nets[i].compile(optimizer=SGD(learning_rate=learningRate), loss='categorical_crossentropy',
                                   metrics=['acc'])
    # 改
    best_local_nets[i].load_weights(filepath + "bestlocal_nets[" + str(i) + "].h5")
    results = best_local_nets[i].evaluate(f_clientDataTest[i], f_clientLabelTest[i])
    y_pred = best_local_nets[i].predict_classes(f_clientDataTest[i])
    y_test = f_clientLabelTest[i].copy()
    if (modelType == 'LSTM' or modelType == 'GRU'):
        y_test = np.argmax(y_test, axis=1)
    macroVal_precision = precision_score(y_test, y_pred, average='macro')
    macroVal_recall = recall_score(y_test, y_pred, average='macro')
    _macroVal_f1 = f1_score(y_test, y_pred, average='macro')

    # 为了画aoc曲线设置！！！
    if (modelType == 'LSTM' or modelType == 'GRU'):
        clientLabelTest[i] = np.argmax(clientLabelTest[i], axis=1)
    classType = []
    for i in range(activityCount):
        classType.append(i)
    y_test = label_binarize(y_test, classes=classType)
    y_pred = label_binarize(y_pred, classes=classType)
    # y_score未知！！！！
    fpr = dict()
    tpr = dict()

    for i in range(activityCount):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())


    #   Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(activityCount)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(activityCount):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= activityCount

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr


    indiAccTest.append(results[1])
    indiMacroPrecision.append(macroVal_precision)
    indiMacroRecall.append(macroVal_recall)
    indiMacroTest.append(_macroVal_f1)
    if (savedClientModel == 1):
        best_local_nets[i].save(filepath + 'models/clientModel' + str(i + 1) + '.h5')
m_indiAccTest = np.mean(indiAccTest)
m_indiMacroTest = np.mean(indiMacroTest)
m_indiMacroPrecision = np.mean(indiMacroPrecision)
m_indiMacroRecall = np.mean(indiMacroRecall)

for i in range(len(best_local_nets)):
    indiAccTest[i] = round(indiAccTest[i],4)
    indiMacroTest[i] = round(indiMacroTest[i],4)
    indiMacroPrecision[i] = round(indiMacroPrecision[i],4)
    indiMacroRecall[i] = round(indiMacroRecall[i],4)

modelStatistics = {
    "Results on individual client models on their own tests": '',
    "mean accuracy:": roundNumber(m_indiAccTest),
    "mean macro f1:": roundNumber(m_indiMacroTest),
    "mean macro precision:": roundNumber(m_indiMacroPrecision),
    "mean macro recall:": roundNumber(m_indiMacroRecall),
    'accuracy':indiAccTest,
    "macro f1:": indiMacroTest,
    "macro precision:": indiMacroPrecision,
    "macro recall:": indiMacroRecall,


}
with open(filepath + 'indivualClientsMeasure.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(modelStatistics.items())


print("Training finished")
