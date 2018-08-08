import numpy as np
from preprocess import standardize

DEBUG = False

class DataGenerator(object):
    def __init__(self, data, windowSize=128, lookback=60, sampleRate=6, prediction=1, normalize=True):
        """ Yield a time frame of data from a larger dataset \n
        
        data {array} - a 1-D numpy array of time-series data, assume values has been standardized\n
        windowSize {int} - indicating the size of the chunk of data to return \n
        lookback {int} - number of previous time steps to keep at each iteration \n
        sampleRate {int} - the period, in timesteps, at which data is sample. \n
        prediction {int} - predict how many future steps """
        if normalize: self.data = standardize(data)
        else: self.data = data
        self.datasize = data.shape[0]
        self.windowSize = windowSize
        self.lookback = lookback
        self.sampleRate = sampleRate
        self.prediction = prediction

        self.maxStepIndex = (sampleRate * prediction) + windowSize
        self.minIndex = int(1 + (self.datasize - (windowSize * lookback)) / sampleRate)
        if self.minIndex < 0 or self.maxStepIndex > self.datasize:
            raise ValueError("Oversize windowSize or lookback")

        #       Samples                target
        #   t-2   t-1    t           t+1    t+2
        # [[w11,  w12,  w13]       [[w11,   w12]
        #  ...                      ...
        #  [wn1,  wn2,  wn3]]       [wn1,   wn2]]

        self.input = np.zeros((windowSize, lookback), dtype=float)
        self.label = np.zeros((windowSize, prediction), dtype=float)

    def generator(self, returnLabel=True):
        index = self.minIndex
        while True:
            if (index + self.maxStepIndex) > self.datasize:
                index = self.minIndex

            # self.input[:,self.lookback] = self.data[index: index + self.windowSize] # Store current timestep

            prevStepIndex = index - (self.sampleRate * self.lookback)
            for prevStep in range(self.lookback):
                # Store n previous timestep at the nth column
                self.input[:,prevStep] = self.data[prevStepIndex:prevStepIndex + self.windowSize]
                prevStepIndex += self.sampleRate
           
            if returnLabel:
                predictStepIndex = index + self.sampleRate
                for predictStep in range(self.prediction):
                    # Store timestep n as at the nth column
                    # print(f"predictStepIndex {predictStepIndex}, ")
                    self.label[:,predictStep] = self.data[predictStepIndex:predictStepIndex + self.windowSize] 
                    predictStepIndex += self.sampleRate
                # LSTM expects input with shape of [samples, time steps, features].
                # Since we are predicting temperature with previous temperature
                # we use the previous timestep as features and keep time steps as 1
                # Input shape will become (windowSize, 1, previousTimeStep)
                yield self.input.reshape(self.windowSize, 1, self.lookback), self.label.copy()
            else:
                yield self.input.reshape(self.windowSize, 1, self.lookback)
            index += self.sampleRate
 

if DEBUG:
    import time
    from datasetAnalysis import readDataset
    
    BENCHMARK_ITERATION = 100000
    dataset = readDataset('./data/train/*.csv')
    # dataset = standardize(dataset)
    # g = DataGenerator(dataset, windowSize=3, lookback=3, sampleRate=1, prediction=4).generator(returnLabel=True)
    g = DataGenerator(dataset, windowSize=128, lookback=24, sampleRate=6, prediction=1).generator(returnLabel=True)
    print(next(g)[1])
    print(next(g)[1])
    print(next(g)[1])
    start = time.time()
    for _ in range(BENCHMARK_ITERATION):
        next(g)
    end = time.time()
    print(f"Total time for iterating {BENCHMARK_ITERATION} times: {end - start:.3f}")

    