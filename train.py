import re 
from rnn import RNN
from datagen import DataGenerator
from preprocess import setup, TRAINING_DATASIZE, VALIDATION_DATASIZE
from datasetAnalysis import readDataset 
from time import gmtime, strftime, time

# Training Hyperparameters
EPOCHS = 20
WINDOW_SIZE = 128
LERNING_RATE = 0.001
SAMPLERATE = 128
LOOKBACK = 144
PREDICTION = 1
STEPS_PER_EPOCH = TRAINING_DATASIZE / WINDOW_SIZE / SAMPLERATE # Calculate steps per epoch based on window size and epochs
VALIDATION_STEP_PER_EPOCH = VALIDATION_DATASIZE / WINDOW_SIZE / SAMPLERATE

# Model parameters
HIDDEN_NODES = 32
LOSS_FUNC = 'mean_squared_error'
OPTIMIZER = 'adam'

# Configuring stored models
BASE_PATH = './models/'
WEIGHT_EXT = 'h5'
ARCHITECT_EXT = 'json'
FILENAME_REGEX = r"hid(\d+)_lkb(\d+)_win(\d+)_smpr(\d+)_pred(\d+)" # Example matches "hid32_lkb144_win28_smpr6_pred1_18-08-08_20:36:25.h5"

def constructFilename(basePath, hiddenNodes, lookback, windowSize, samplerate, prediction, fileExt):
    """ Generate a unique filename for storing the model weights \n
        
        basePath {str} - base path to store the file \n
        hiddenNodes {int} - number of hidden nodes the model contains \n
        lookback {int} - number of previous time steps the model refer to \n
        windowSize {int} - size of data sample the model can process per time step \n
        fileExt {str} - the stored file extension \n
        
        Returns: a templated filename """
    currTime = strftime("%y-%m-%d_%H-%M-%S", gmtime())
    return f"{basePath}hid{hiddenNodes}_lkb{lookback}_win{windowSize}_smpr{samplerate}_pred{prediction}_{currTime}.{fileExt}"

def main(save=True):
    """ Train a model \n
        
        ave {bool} - whether to save the trained model (default: True) \n
        
        Returns: wrapper RNN class for a Keras model (e.g. keras.models.Sequential) """
    startTime = time()
    trainingSet, validationSet, scaler = setup()
    trainGen = DataGenerator(trainingSet, scaler, windowSize=WINDOW_SIZE, lookback=LOOKBACK, 
                sampleRate=SAMPLERATE, prediction=PREDICTION).generator()
    validGen = DataGenerator(validationSet, scaler, windowSize=WINDOW_SIZE, lookback=LOOKBACK,
                sampleRate=SAMPLERATE, prediction=PREDICTION).generator()
    rnn = RNN(HIDDEN_NODES, LOOKBACK, WINDOW_SIZE, SAMPLERATE, PREDICTION)
    optimizer = rnn.pickOptimizer(OPTIMIZER, lr=LERNING_RATE)
    rnn.model.compile(loss=LOSS_FUNC, optimizer=optimizer)
    rnn.model.fit_generator(
            trainGen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
            validation_data=validGen, validation_steps=VALIDATION_STEP_PER_EPOCH,
            verbose=2, shuffle=False
        )
    endTime = time()
    print(f"\nTRAINING DONE. Total time elapsed: {strftime('%H:%M:%S', gmtime(endTime - startTime))}")
    if save:
        weightsFile = constructFilename(BASE_PATH, HIDDEN_NODES, LOOKBACK, WINDOW_SIZE, SAMPLERATE, PREDICTION, WEIGHT_EXT)
        architectureFile = constructFilename(BASE_PATH, HIDDEN_NODES, LOOKBACK, WINDOW_SIZE, SAMPLERATE, PREDICTION, ARCHITECT_EXT)
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
    hiddenNodes, lookback, windowSize, samplerate, prediction = parseFilename(filename)
    rnn = RNN(int(hiddenNodes), int(lookback), int(windowSize), int(samplerate), int(prediction))
    rnn.load(filename)
    return rnn

if __name__ == '__main__':
    main(True)
    # loadTrainedModel(BASE_PATH + 'hidden5_lookback5_window128_08-06-17:18:15.h5')