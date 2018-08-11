import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from rnn import RNN
from datasetAnalysis import readDataset, TRAINING_SET, VALIDATION_SET, TEST_SET, TEST_SET_STAT
from datagen import DataGenerator
from preprocess import setup, TRAINING_DATASIZE, VALIDATION_DATASIZE
from train import loadTrainedModel, HIDDEN_NODES, LOOKBACK, WINDOW_SIZE, SAMPLERATE, PREDICTION

DEBUG = True

# TRAINED_MODEL = './models/working/hid32_lkb144_win128_smpr6_pred1_18-08-09_02-29-57.h5'
TRAINED_MODEL = './models/hid32_lkb144_win128_smpr128_pred1_18-08-11_16-29-35.h5'

# Loading pre-calculated statistics on the test data
with open(TEST_SET_STAT, 'rb') as f: stat = pickle.load(f)
TEST_DATASIZE = stat["shape"][0]

def take(n, iterable):
    "Return first n items of the iterable as a numpy array"
    return np.block(list(islice(iterable, n))).ravel()

def predict(modelpath, UNTRAINED_MODEL=False):
    if UNTRAINED_MODEL: rnn = RNN(HIDDEN_NODES, LOOKBACK, WINDOW_SIZE, SAMPLERATE, 1)
    else: rnn = loadTrainedModel(modelpath)

    trainingSet, validationSet, scaler = setup()
    testSet = readDataset(TEST_SET)

    if rnn.sampleRate < rnn.windowSize:
        trainGen = DataGenerator(trainingSet, scaler, windowSize=rnn.windowSize, 
                                lookback=rnn.lookBack, sampleRate=rnn.windowSize)
        validateGen = DataGenerator(validationSet, scaler, windowSize=rnn.windowSize, 
                                lookback=rnn.lookBack, sampleRate=rnn.windowSize)
        testGen = DataGenerator(testSet, scaler, windowSize=rnn.windowSize, 
                                lookback=rnn.lookBack, sampleRate=rnn.windowSize)
        batchLength = rnn.windowSize

    else:
        trainGen = DataGenerator(trainingSet, scaler, windowSize=rnn.windowSize, 
                                lookback=rnn.lookBack, sampleRate=rnn.sampleRate)
        validateGen = DataGenerator(validationSet, scaler, windowSize=rnn.windowSize, 
                                lookback=rnn.lookBack, sampleRate=rnn.sampleRate)
        testGen = DataGenerator(testSet, scaler, windowSize=rnn.windowSize, 
                                lookback=rnn.lookBack, sampleRate=rnn.sampleRate)
        batchLength = rnn.sampleRate # or sampleRate * windowSize?

    trainingSetTrueSize = TRAINING_DATASIZE - trainGen.maxStepIndex - trainGen.minIndex
    validationSetTrueSize = VALIDATION_DATASIZE - validateGen.maxStepIndex - validateGen.minIndex
    testSetTrueSize = TEST_DATASIZE - testGen.maxStepIndex - testGen.minIndex
    trainStep = int(trainingSetTrueSize / batchLength)
    validateStep = int(validationSetTrueSize / batchLength)
    testStep = int(testSetTrueSize / batchLength)
    
    if DEBUG: print(f"trainStep: {trainStep}, validationStep: {validateStep}, testStep: {testStep}")

    # Model predictions
    start = time.time()
    trainPred = rnn.model.predict_generator(trainGen.generator(returnLabel=False), trainStep)
    end = time.time()
    if DEBUG: print(f"Time to make {trainPred.shape} training predictions: {end - start:.3f}, training dataset shape {trainingSet.shape}")
    
    start = time.time()
    validatePred = rnn.model.predict_generator(validateGen.generator(returnLabel=False), validateStep)
    end = time.time()
    if DEBUG: print(f"Time to make {validatePred.shape} validation predictions: {end - start:.3f}, validation dataset shape {validationSet.shape}")
    
    start = time.time()
    testPred = rnn.model.predict_generator(testGen.generator(returnLabel=False), testStep)
    end = time.time()
    if DEBUG: print(f"Time to make {testPred.shape} test predictions: {end - start:.3f}, test dataset shape {testSet.shape}")
    
    # Undo the standardization on the predictions
    trainPred = scaler.inverse_transform(trainPred)
    validatePred = scaler.inverse_transform(validatePred)
    testPred = scaler.inverse_transform(testPred)

    #  Sampling like this
    #   | - minIndex - |                 | - maxStepIndex - |
    #  [   ..........  {    TRUE SIZE    }  ..............  ]
    trainingTruth = trainingSet[trainGen.minIndex:-trainGen.maxStepIndex].ravel()
    validationTruth = validationSet[validateGen.minIndex:-validateGen.maxStepIndex].ravel()
    testTruth = testSet[testGen.minIndex:-testGen.maxStepIndex].ravel()
    
    if DEBUG: print(f"trainingTruth shape: {trainingTruth.shape}, validationTruth shape: {validationTruth.shape}, testTruth shape: {testTruth.shape}")
    
    groundTruth = np.block([trainingTruth, validationTruth, testTruth])

    return trainPred, validatePred, testPred, groundTruth

def plotPrediction(trainPred, validatePred, testPred, groundTruth):
    validatePredPlot = np.empty_like(groundTruth)
    validatePredPlot.fill(np.nan)
    startIndex = trainPred.shape[0]
    endIndex = startIndex + validatePred.shape[0]
    validatePredPlot[startIndex : endIndex] = validatePred.ravel()

    testPredPlot = np.empty_like(groundTruth)
    testPredPlot.fill(np.nan)
    startIndex = trainPred.shape[0] + validatePred.shape[0]
    endIndex = startIndex + testPred.shape[0] 
    testPredPlot[startIndex:endIndex] = testPred.ravel()
    plt.plot(groundTruth)
    # pred = np.block([trainPred, validatePred, testPred])
    # plt.plot(pred)
    plt.plot(trainPred, alpha=0.5)
    plt.plot(validatePredPlot, alpha=0.5)
    plt.plot(testPredPlot, alpha=0.5)
    plt.show()

if __name__ == '__main__':
    trainPred, validatePred, testPred, groundTruth = predict(TRAINED_MODEL)
    plotPrediction(trainPred, validatePred, testPred, groundTruth)