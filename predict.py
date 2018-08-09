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

TRAINED_MODEL = './models/working/hid32_lkb144_win128_smpr6_pred1_18-08-09_02-29-57.h5'

# Loading pre-calculated statistics on the test data
with open(TEST_SET_STAT, 'rb') as f: stat = pickle.load(f)
TEST_DATASIZE = stat["shape"][0]

def take(n, iterable):
    "Return first n items of the iterable as a numpy array"
    return np.block(list(islice(iterable, n))).ravel()

def predict(UNTRAINED_MODEL=False):
    if UNTRAINED_MODEL: rnn = RNN(HIDDEN_NODES, LOOKBACK, WINDOW_SIZE, SAMPLERATE, 1)
    else: rnn = loadTrainedModel(TRAINED_MODEL)

    trainingSet, validationSet, scaler = setup()
    testSet = readDataset(TEST_SET)
    
    trainGen = DataGenerator(trainingSet, scaler, windowSize=rnn.windowSize, 
                            lookback=rnn.lookBack, sampleRate=SAMPLERATE)
    validateGen = DataGenerator(validationSet, scaler, windowSize=rnn.windowSize, 
                            lookback=rnn.lookBack, sampleRate=SAMPLERATE)
    testGen = DataGenerator(testSet, scaler, windowSize=rnn.windowSize, 
                            lookback=rnn.lookBack, sampleRate=SAMPLERATE)
    
    trainStep = int((TRAINING_DATASIZE - trainGen.maxStepIndex - trainGen.minIndex) / SAMPLERATE)
    validateStep = int((VALIDATION_DATASIZE - validateGen.maxStepIndex - validateGen.minIndex) / SAMPLERATE)
    testStep = int((TEST_DATASIZE - testGen.maxStepIndex - testGen.minIndex) / SAMPLERATE)
    
    if DEBUG: print(f"trainStep: {trainStep}, validationStep: {validateStep}, testStep: {testStep}")

    # Model predictions
    start = time.time()
    trainPred = rnn.model.predict_generator(trainGen.generator(returnLabel=False), trainStep)
    end = time.time()
    if DEBUG: print(f"Time to make {trainPred.shape[0]} training predictions: {end - start:.3f}, shape {trainPred.shape}")
    
    start = time.time()
    validatePred = rnn.model.predict_generator(validateGen.generator(returnLabel=False), validateStep)
    end = time.time()
    if DEBUG: print(f"Time to make {validatePred.shape[0]} validation predictions: {end - start:.3f}, shape {validatePred.shape}")
    
    start = time.time()
    testPred = rnn.model.predict_generator(testGen.generator(returnLabel=False), testStep)
    end = time.time()
    if DEBUG: print(f"Time to make {testPred.shape[0]} test predictions: {end - start:.3f}, shape {testPred.shape}")
    
    # Undo the standardization on the predictions
    trainPred = scaler.inverse_transform(trainPred)
    validatePred = scaler.inverse_transform(validatePred)
    testPred = scaler.inverse_transform(testPred)

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
    # testPredPlot[startIndex:] = testPred[:testPredPlot.shape[0] - startIndex].ravel() # TODO: debug this
    testPredPlot[startIndex:endIndex] = testPred.ravel()
    plt.plot(groundTruth)
    plt.plot(trainPred, alpha=0.5)
    plt.plot(validatePredPlot, alpha=0.5)
    plt.plot(testPredPlot, alpha=0.5)
    plt.show()


# def getGroundTruth():
#     trainingSet = readDataset(TRAINING_SET)
#     validationSet = readDataset(VALIDATION_SET)
#     testSet = readDataset(TEST_SET)
#     return np.block([trainingSet, validationSet, testSet])

if __name__ == '__main__':
    trainPred, validatePred, testPred, groundTruth = predict()
    plotPrediction(trainPred, validatePred, testPred, groundTruth)