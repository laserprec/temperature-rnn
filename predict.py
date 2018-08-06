import glob
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from rnn import RNN
from datasetAnalysis import readDataset, TRAINING_SET, VALIDATION_SET, TEST_SET, TEST_SET_STAT
from preprocess import datasetGenerator, unstandardize, TRAINING_DATASIZE, VALIDATION_DATASIZE
from train import train, loadTrainedModel, TRAINING_SET, HIDDEN_NODES, LOOKBACK, WINDOW_SIZE
from datasetAnalysis import readDataset

DEBUG = True
UNTRAINED_MODEL = True

TRAINED_MODEL = './models/hidden5_lookback5_window128_18-08-06_17:18:15.h5'

# Loading pre-calculated statistics on the test data
with open(TEST_SET_STAT, 'rb') as f: stat = pickle.load(f)
TEST_DATASIZE = stat["shape"][0]

def predict():
    if UNTRAINED_MODEL: rnn = RNN(HIDDEN_NODES, LOOKBACK, WINDOW_SIZE)
    else: rnn = loadTrainedModel(TRAINED_MODEL)

    trainGen = datasetGenerator(TRAINING_SET, rnn.windowSize, rnn.lookBack, inputOnly=True)
    validateGen = datasetGenerator(VALIDATION_SET, rnn.windowSize, rnn.lookBack, inputOnly=True)
    testGen = datasetGenerator(TEST_SET, rnn.windowSize, rnn.lookBack, inputOnly=True)
    
    trainStep = TRAINING_DATASIZE / rnn.windowSize
    validateStep = VALIDATION_DATASIZE / rnn.windowSize
    testStep = TEST_DATASIZE / rnn.windowSize

    # Model predictions
    start = time.time()
    trainPred = rnn.model.predict_generator(trainGen, trainStep)
    end = time.time()
    if DEBUG: print(f"Time to make {trainPred.shape[0]} training predictions: {end - start:.3f}")
    
    start = time.time()
    validatePred = rnn.model.predict_generator(validateGen, validateStep)
    end = time.time()
    if DEBUG: print(f"Time to make {validatePred.shape[0]} validation predictions: {end - start:.3f}")
    
    start = time.time()
    testPred = rnn.model.predict_generator(testGen, testStep)
    end = time.time()
    if DEBUG: print(f"Time to make {testPred.shape[0]} test predictions: {end - start:.3f}")

    if DEBUG: print(f"Training Prediction stats: mean: {np.mean(trainPred):.3f}, std: {np.std(trainPred):.3f}" +
        f" max: {trainPred.max():.3f} min: {trainPred.min():.3f} ")
    # Undo the standardization on the predictions
    trainPred = unstandardize(trainPred)
    validatePred = unstandardize(validatePred)
    testPred = unstandardize(testPred)

    return trainPred, validatePred, testPred

def plotPrediction(trainPred, validatePred, testPred, groundTruth):
    validatePredPlot = np.empty_like(groundTruth)
    validatePredPlot.fill(np.nan)
    startIndex = trainPred.shape[0]
    endIndex = startIndex + validatePred.shape[0]
    validatePredPlot[startIndex : endIndex] = validatePred.ravel()

    testPredPlot = np.empty_like(groundTruth)
    testPredPlot.fill(np.nan)
    startIndex = trainPred.shape[0] + validatePred.shape[0]
    # endIndex = startIndex + testPred.shape[0]
    # print(f"ground truth length {groundTruth.shape}, startIndex {startIndex}, endIndex {endIndex}")
    testPredPlot[startIndex:] = testPred[:testPredPlot.shape[0] - startIndex].ravel()
    plt.plot(groundTruth)
    plt.plot(trainPred, alpha=0.5)
    plt.plot(validatePredPlot, alpha=0.5)
    plt.plot(testPredPlot, alpha=0.5)
    plt.show()


def getGroundTruth():
    trainingSet = glob.glob(TRAINING_SET)
    validationSet = glob.glob(VALIDATION_SET)
    testSet = glob.glob(TEST_SET)
    dataset = trainingSet + validationSet + testSet
    return readDataset(dataset)

if __name__ == '__main__':
    trainPred, validatePred, testPred = predict()
    groundTruth = getGroundTruth()
    plotPrediction(trainPred, validatePred, testPred, groundTruth)