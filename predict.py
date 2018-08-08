import glob
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from rnn import RNN
from datasetAnalysis import readDataset, TRAINING_SET, VALIDATION_SET, TEST_SET, TEST_SET_STAT
from datagen import DataGenerator
from preprocess import standardize, unstandardize, TRAINING_DATASIZE, VALIDATION_DATASIZE
from train import train, loadTrainedModel, HIDDEN_NODES, LOOKBACK, WINDOW_SIZE, SAMPLERATE
from datasetAnalysis import readDataset

DEBUG = True

TRAINED_MODEL = './models/hidden32_lookback24_window128_18-08-08_19:30:37.h5'

# Loading pre-calculated statistics on the test data
with open(TEST_SET_STAT, 'rb') as f: stat = pickle.load(f)
TEST_DATASIZE = stat["shape"][0]

def predict(UNTRAINED_MODEL=False):
    if UNTRAINED_MODEL: rnn = RNN(HIDDEN_NODES, LOOKBACK, WINDOW_SIZE)
    else: rnn = loadTrainedModel(TRAINED_MODEL)

    trainingSet = readDataset(TRAINING_SET)
    validationSet = readDataset(VALIDATION_SET)
    testSet = readDataset(TEST_SET)
    
    trainGen = DataGenerator(trainingSet, rnn.windowSize, rnn.lookBack, SAMPLERATE).generator(returnLabel=False)
    validateGen = DataGenerator(validationSet, rnn.windowSize, rnn.lookBack, SAMPLERATE).generator(returnLabel=False)
    testGen = DataGenerator(testSet, rnn.windowSize, rnn.lookBack, SAMPLERATE).generator(returnLabel=False)
    
    trainStep = TRAINING_DATASIZE / rnn.windowSize
    validateStep = VALIDATION_DATASIZE / rnn.windowSize
    testStep = TEST_DATASIZE / rnn.windowSize

    # Model predictions
    start = time.time()
    trainPred = rnn.model.predict_generator(trainGen, trainStep)
    end = time.time()
    if DEBUG: print(f"Time to make {trainPred.shape[0]} training predictions: {end - start:.3f}, shape {trainPred.shape}")
    
    start = time.time()
    validatePred = rnn.model.predict_generator(validateGen, validateStep)
    end = time.time()
    if DEBUG: print(f"Time to make {validatePred.shape[0]} validation predictions: {end - start:.3f}, shape {validatePred.shape}")
    
    start = time.time()
    testPred = rnn.model.predict_generator(testGen, testStep)
    end = time.time()
    if DEBUG: print(f"Time to make {testPred.shape[0]} test predictions: {end - start:.3f}, shape {testPred.shape}")

    # if DEBUG: print(f"Training Prediction stats: mean: {np.mean(trainPred):.3f}, std: {np.std(trainPred):.3f}" +
    #     f" max: {trainPred.max():.3f} min: {trainPred.min():.3f} ")

    # Undo the standardization on the predictions
    trainPred = unstandardize(trainPred)
    validatePred = unstandardize(validatePred)
    testPred = unstandardize(testPred)

    groundTruth = np.block([trainingSet, validationSet, testSet])

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
    # endIndex = startIndex + testPred.shape[0]
    # print(f"ground truth length {groundTruth.shape}, startIndex {startIndex}, endIndex {endIndex}")
    testPredPlot[startIndex:] = testPred[:testPredPlot.shape[0] - startIndex].ravel() # TODO: debug this
    # testPredPlot[startIndex:] = testPred.ravel()
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