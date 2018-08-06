import re
import glob
from rnn import RNN
from preprocess import datasetGenerator, TRAINING_DATASIZE, VALIDATION_DATASIZE
from time import gmtime, strftime

TRAINING_SET = './data/train/*.csv'
VALIDATION_SET = './data/validate/*.csv'

# Training Hyperparameters
EPOCHS = 20
WINDOW_SIZE = 128
STEPS_PER_EPOCH = TRAINING_DATASIZE / WINDOW_SIZE  # Calculate steps per epoch based on window size and epochs
VALIDATION_STEP_PER_EPOCH = VALIDATION_DATASIZE / WINDOW_SIZE

# Model parameters
HIDDEN_NODES = 5
LOOKBACK = 5
LOSS_FUNC = 'mean_squared_error'
OPTIMIZER = 'adam'

# Configuring stored models
BASE_PATH = './models/'
WEIGHT_EXT = 'h5'
ARCHITECT_EXT = 'json'
FILENAME_REGEX = r"hidden(\d+)_lookback(\d+)_window(\d+)" # Example matches "hidden5_lookback2_window10_08-06-15:36:39.h5"

def dataSplit():
    return TRAINING_SET, VALIDATION_SET

def constructFilename(basePath, hiddenNodes, lookback, windowSize, fileExt):
    """ Generate a unique filename for storing the model weights \n
        
        basePath {str} - base path to store the file \n
        hiddenNodes {int} - number of hidden nodes the model contains \n
        lookback {int} - number of previous time steps the model refer to \n
        windowSize {int} - size of data sample the model can process per time step \n
        fileExt {str} - the stored file extension \n
        
        Returns: a templated filename """
    currTime = strftime("%y-%m-%d_%H:%M:%S", gmtime())
    return f"{basePath}hidden{hiddenNodes}_lookback{lookback}_window{windowSize}_{currTime}.{fileExt}"

def train(save=True):
    """ Train a model \n
        
        ave {bool} - whether to save the trained model (default: True) \n
        
        Returns: wrapper RNN class for a Keras model (e.g. keras.models.Sequential) """
    trainingSet, validationSet = dataSplit()
    trainGen = datasetGenerator(trainingSet, WINDOW_SIZE, LOOKBACK)
    validGen = datasetGenerator(validationSet, WINDOW_SIZE, LOOKBACK)
    rnn = RNN(HIDDEN_NODES, LOOKBACK, WINDOW_SIZE)
    rnn.model.compile(loss=LOSS_FUNC, optimizer=OPTIMIZER)
    rnn.model.fit_generator(
            trainGen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
            validation_data=validGen, validation_steps=VALIDATION_STEP_PER_EPOCH,
            verbose=2, shuffle=False
        )
    if save:
        weightsFile = constructFilename(BASE_PATH, HIDDEN_NODES, LOOKBACK, WINDOW_SIZE, WEIGHT_EXT)
        architectureFile = constructFilename(BASE_PATH, HIDDEN_NODES, LOOKBACK, WINDOW_SIZE, ARCHITECT_EXT)
        rnn.saveWeights(weightsFile)
        rnn.saveArchitecture(architectureFile)
    return rnn

def parseFilename(filename):
    """ filename {str} - a templated filename storing the trained weights \n
        
        Returns: a tuple of model-related info (e.g number of hiddenNodes and lookback) """
    parser = re.compile(FILENAME_REGEX)
    return parser.findall(filename)[0]

def loadTrainedModel(filename):
    """ Parse model-related info from its filename \n
        
        filename {str} - a templated filename storing the trained weights \n
        
        Returns: wrapper RNN class for a Keras model (e.g. keras.models.Sequential) """
    hiddenNodes, lookback, windowSize = parseFilename(filename)
    rnn = RNN(int(hiddenNodes), int(lookback), int(windowSize))
    rnn.load(filename)
    return rnn

if __name__ == '__main__':
    train(True)
    # loadTrainedModel(BASE_PATH + 'hidden5_lookback5_window128_08-06-17:18:15.h5')