import warnings
import numpy as np

DEBUG = False

class DataGenerator(object):
    def __init__(self, data, scaler, windowSize=128, lookback=60,
                    sampleRate=6, prediction=1, numFeatures=1, normalize=True):
        """ Yield a time frame of data from a larger dataset \n
        
        data {array} - a 1-D numpy array of time-series data, assume values has been standardized\n
        scaler {obj} - a sklearn.preprocessing.MinMaxScaler complied on training data
        windowSize {int} - indicating the size of the chunk of data to return \n
        lookback {int} - number of previous time steps to keep at each iteration \n
        sampleRate {int} - the period, in timesteps, at which data is sample. \n
        prediction {int} - predict how many future steps """
        if normalize: self.data = scaler.transform(data)
        else: self.data = data
        self.scaler = scaler
        self.datasize = data.shape[0]
        self.windowSize = windowSize
        self.lookback = lookback
        self.sampleRate = sampleRate
        self.prediction = prediction
        self.numFeatures = numFeatures

        self.maxStepIndex = (sampleRate * prediction) + windowSize
        if sampleRate < windowSize:
            self.minIndex = int(1 + (self.datasize - (windowSize * lookback)) / sampleRate)
        else:
            self.minIndex = (sampleRate * lookback) + windowSize
        if self.minIndex < 0 or self.minIndex > self.datasize:
            raise ValueError("Oversize lookback or sampleRate")
        if self.maxStepIndex > self.datasize:
            raise ValueError("Oversize prediction or sampleRate")

        #       Samples                target
        #   t-2   t-1    t           t+1    t+2
        # [[w11,  w12,  w13]       [[w11,   w12]
        #  ...                      ...
        #  [wn1,  wn2,  wn3]]       [wn1,   wn2]]

        self.input = np.zeros((windowSize, lookback), dtype=float)
        self.label = np.zeros((windowSize, prediction), dtype=float)

    def generator(self, returnInput=True, returnLabel=True):
        index = self.minIndex
        while True:
            if (index + self.maxStepIndex) > self.datasize:
                index = self.minIndex

            if returnInput:
                prevStepIndex = index - (self.sampleRate * self.lookback)
                for prevStep in range(self.lookback):
                    # Store n previous timestep at the nth column
                    self.input[:,prevStep] = self.data[prevStepIndex:prevStepIndex + self.windowSize].ravel()
                    prevStepIndex += self.sampleRate
           
            if returnLabel:
                predictStepIndex = index + self.sampleRate
                for predictStep in range(self.prediction):
                    # Store timestep n as at the nth column
                    # print(f"predictStepIndex {predictStepIndex}, ")
                    self.label[:,predictStep] = self.data[predictStepIndex:predictStepIndex + self.windowSize].ravel()
                    predictStepIndex += self.sampleRate

            index += self.sampleRate

            if returnInput and returnLabel:
                # LSTM expects input with shape of [samples, time steps, features].
                yield self.input.reshape(self.windowSize, self.lookback, self.numFeatures), self.label.copy()
            elif returnInput:
                yield self.input.reshape(self.windowSize, self.lookback, self.numFeatures)
            elif returnLabel:
                yield self.label.copy()
            else:
                warnings.warn("Data generator not returning anything")
                yield None

            
 

if DEBUG:
    import time
    from datasetAnalysis import readDataset
    
    BENCHMARK_ITERATION = 100000
    dataset = readDataset('./data/train/*.csv')
    # dataset = standardize(dataset)
    # g = DataGenerator(dataset, windowSize=3, lookback=3, sampleRate=1, prediction=4).generator(returnLabel=True)
    g = DataGenerator(dataset, None, windowSize=10, lookback=5, sampleRate=6, prediction=1, normalize=False).generator(returnLabel=True)
    print(next(g)[1])
    print(next(g)[1])
    print(next(g)[1])
    # start = time.time()
    # for _ in range(BENCHMARK_ITERATION):
    #     next(g)
    # end = time.time()
    # print(f"Total time for iterating {BENCHMARK_ITERATION} times: {end - start:.3f}")

    