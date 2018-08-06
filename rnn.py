# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class RNN(object):
    def __init__(self, hiddenNodes, lookBack, windowSize):
        self.hiddenNodes = int(hiddenNodes)
        self.lookBack = int(lookBack)
        self.windowSize = int(windowSize)

        self.model = Sequential()
        self.model.add(LSTM(hiddenNodes, input_shape=(lookBack, 1)))
        self.model.add(Dense(1))
        print(self.model.summary(), '\n')

    def saveWeights(self, filename):
        self.model.save_weights(filename)
        print(f"\n SAVING model weights to {filename}")

    def saveArchitecture(self, filename):
        with open(filename, 'w') as fp: fp.write(self.model.to_json())
        print(f"SAVING model architecture to {filename} \n")

    def load(self, filename):
        try:
            self.model.load_weights(filename)
            print(f"\nLOADING model weights from {filename}\n")
        except OSError:
            print(f"\n ERROR: No such file \"{filename}\" \n")
            exit


# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)