import csv
import glob
import pickle
import numpy as np
from copy import deepcopy
from itertools import tee, islice
from datetime import datetime

DEBUG = False

# Loading pre-calculated statistics on the training data
TRAINING_SET_STAT = './data/train/stat.pickle'
with open(TRAINING_SET_STAT, 'rb') as f: stat = pickle.load(f)
TRAINING_STD = stat["std"]
TRAINING_MEAN = stat["mean"]
TRAINING_DATASIZE = stat["shape"][0]

# Loading pre-calculated statistics on the validation data
VALIDATION_SET_STAT = './data/validate/stat.pickle'
with open(VALIDATION_SET_STAT, 'rb') as f: stat = pickle.load(f)
VALIDATION_DATASIZE = stat["shape"][0]

def parseDate(dateStr):
    """
    Replaces datetime.strptime() for performance gain
    """
    return datetime(
                int(dateStr[6:10]),  #year
                int(dateStr[3:5]),   #month
                int(dateStr[:2]),    #day
                int(dateStr[11:13]), #hour
                int(dateStr[14:16]), #minute
                int(dateStr[17:19]), #second
                )

def unstandardize(data):
    return data * TRAINING_STD + TRAINING_MEAN

def standardize(data):
    return (data - TRAINING_MEAN) / TRAINING_STD

def dataGenerator(filenames, windowSize=100, shiftIndex=1):
    """ Generate each row in a list of .csv files \n
        
        filenames {list} - a list of filenames to read data from \n
        windowSize {int} - indicating the size of the chunk of data to return \n
        shiftIndex {int} - the amount of space the data window shifts at each iteration \n
        std {float}      - the standard deviation of the overall collection of the read data,
                           used to standardize the data value \n
        mean {float}    - the mean of the overall collection of read data,
                           used to standardize the data value 
        
        Returns: iterable"""
    temp = np.zeros(windowSize, dtype=float)
    while True:
        for filename in filenames:
            with open(filename, newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)                      # skip reading the header row
                index = 0
                for row in reader:
                    temperature = np.float(row[2])
                    if -100 < temperature < 100:          # ignore error measurements
                        temp[index] = standardize(temperature)
                    if index == windowSize - 1:
                        yield temp.copy()
                        if shiftIndex > 0:
                            index = shiftIndex - 1
                            temp[:shiftIndex] = temp[-shiftIndex:]  # keep the look back element in the next patch
                        else: raise ValueError("Invalid shiftIndex value")
                    index += 1

# The LSTM network expects the input data (X) in the form of: 
# [samples, time steps, features].
def datasetGenerator(filepath, windowSize, lookback, inputOnly=False):
    """ Generate data for each time step from a dataset \n
        
        filepath {str} - a path to a folder containing a list of .csv files for the dataset \n
        windowSize {int} - size of the batch of data returned at each iteration \n
        lookback {int} - number of previous time steps to keep at each iteration \n
        std {float} - the standard deviation of the dataset (for standardizing each datapoint) \n
        mean {float} - the mean of the dataset (for standardizing each datapoint) 
        
        Returns: iterable"""
    filenames = glob.glob(filepath)
    datagen = dataGenerator(filenames, windowSize)
    iterators = tee(datagen, lookback + 1)
    shiftedIterators = []
    # shift iterators to start at different iteration in the sequence
    for startIndex in range(lookback + 1):
        shiftedIterators.append(islice(iterators[startIndex], startIndex, None))

    for (*inputs, label) in zip(*shiftedIterators):
        inputs = np.array(inputs).reshape(windowSize, lookback, 1)
        if inputOnly:
            yield inputs
        else:
            yield inputs, label

            
if DEBUG:
    import time
    BENCHMARK_ITERATION = 10000
    g = datasetGenerator('./data/train/*.csv', 5, 20)
    start = time.time()
    for _ in range(BENCHMARK_ITERATION):
        next(g)
    end = time.time()
    print(f"Total time for iterating {BENCHMARK_ITERATION} times: {end - start:.3f}")

    