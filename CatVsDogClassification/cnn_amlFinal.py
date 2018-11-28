import numpy as np
import scipy.io as sio 
from scipy.misc import toimage #
from keras.models import Sequential #Used in building network
from keras.layers import Input, Convolution2D, Activation, MaxPooling2D, Dense, Dropout, Flatten # Building BLOCKS of Network
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD  #Stochastic Gradient Descent for training
from keras.utils import np_utils # categorical conversion
from keras import backend as K

K.set_image_dim_ordering('th') #Since depth is at index 1 in image data, followed by length and width
np.random.seed(7) #For reproducing the result
print('Seed Set to 7...')


#function for building the network
def build_network():
    print('Building network...')
    network=Sequential()

    network.add( Convolution2D(32, 7, 7, border_mode='valid', input_shape=(3,64,64), init='glorot_uniform'))
    network.add(Activation('relu'))
    network.add(Convolution2D(64, 5, 5, init='glorot_uniform'))
    network.add(Activation('relu'))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Dropout(0.25))
    network.add(Convolution2D(128, 3, 3, init='glorot_uniform'))
    network.add(Activation('relu'))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Dropout(0.25))

    network.add(Flatten())
    network.add(Dense(512, init='glorot_uniform'))
    network.add(Activation('relu'))
    network.add(Dropout(0.5))
    network.add(Dense(128, init='glorot_uniform'))
    network.add(Activation('relu'))
    network.add(Dropout(0.5))
    network.add(Dense(2))
    network.add(Activation('softmax'))

    sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    network.load_weights("/home/sanjeev/Stuffs/AML/DogCatdata/weights.hdf5")
    network.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print "Network Built..."

    return network


if __name__ == "__main__":

    batch_size = 64
    nb_epoch = 20
    #Load Training Data
    print "Loading Train data from mat files..."
    traindata = sio.loadmat('traindata.mat')
    trainX = traindata['trainX']
    trainX = np.reshape(trainX,(trainX.shape[0],3,64,64))
    trainX = trainX.astype('float32')
    trainX/=255
    #toimage(trainX[0]).save('yo.png') #1st image is saved as yo.png. By viewing this, we can check that reshaping has been done   properly	
    trainY = traindata['trainY']
    trainY = trainY.astype('int')
    # for use with categorical_crossentropy
    trainY = np_utils.to_categorical(trainY, 2)
    print "Data converted to float32 and normalized as /=255"
    #Building Network Architecture
    network = build_network()
    #Training the model
    checkpointer = ModelCheckpoint(filepath="/home/sanjeev/Stuffs/AML/DogCatdata/weights.hdf5", verbose=1, save_best_only=True) 
    hist = network.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split = 0.3, callbacks=[checkpointer])
    print(hist.history)  
    #Load Testing Data
    print "Loading Train data from mat files..."
    testdata = sio.loadmat('testdata.mat')   
    testX = testdata['testX']
    testX = np.reshape(testX,(testX.shape[0],3,64,64))
    testX = testX.astype('float32')
    testX/=255
    print "Data converted to float32 and normalized as /=255"
    #Predicting the output labels of testing data
    testPredict = network.predict(testX, verbose=1)
    sio.savemat('/home/sanjeev/Stuffs/AML/DogCatdata/testResult2.mat', mdict={'testResult': testPredict})
    print "Data Predicted and saved in testResult2.mat."




