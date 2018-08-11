from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

OPTIMIZERS = {'sgd', 'rmsprop', 'adadelta', 'adamax', 'nadam'}

class RNN(object):
    def __init__(self, hiddenNodes, lookBack, windowSize, sampleRate, prediction):
        self.hiddenNodes = int(hiddenNodes)
        self.lookBack = int(lookBack)
        self.windowSize = int(windowSize)
        self.sampleRate = int(sampleRate)
        self.prediction = int(prediction)

        self.model = Sequential()
        self.model.add(LSTM(hiddenNodes, input_shape=(lookBack, 1)))
        self.model.add(Dense(prediction))
        print(self.model.summary(), '\n')

    def pickOptimizer(self, optimizerName, **kwargs):
        if optimizerName == 'adam':
            return Adam(**kwargs)
        elif optimizerName in OPTIMIZERS:
            return optimizerName
        else:
            raise ValueError(f"Invalid optimizer. Choose one from here {OPTIMIZERS}")

    def saveWeights(self, filename):
        self.model.save_weights(filename)
        print(f"\nSAVING model weights to {filename}")

    def saveArchitecture(self, filename):
        with open(filename, 'w') as fp: fp.write(self.model.to_json())
        print(f"SAVING model architecture to {filename} \n")

    def load(self, filename):
        try:
            self.model.load_weights(filename)
            print(f"\nLOADING model weights from {filename}\n")
            print(f"MODEL Parmeters: {self.hiddenNodes} hiddenNodes, {self.lookBack} lookBack, " +
                  f"{self.windowSize} windowSize, {self.sampleRate} sampleRate, {self.prediction} prediction step")
        except OSError:
            print(f"\n ERROR: No such file \"{filename}\" \n")
            exit

