from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Bidirectional
from keras.layers import recurrent
import numpy as np
import os

import utils.dataprep as dp
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.models import load_model

def generateModel(X_train,Y_train,X_test,Y_test, testDataFiles="",
                  resultFile=""):
    # LSTM model
    # model = Sequential()
    # model.add(recurrent.LSTM(32, input_dim=1, input_length=99,
    #                         activation='sigmoid',
    # inner_activation='hard_sigmoid'))
    # model.add(Dropout(0.5))
    # model.add(recurrent.LSTM(10))

    # model 1
    # model = Sequential()
    # model.add(recurrent.GRU(64, input_dim=2,
    #                                       input_length=100,
    #                          activation='sigmoid',
    #                          inner_activation='hard_sigmoid'))
    # # model.add(Dropout(0.5))
    # # model.add(recurrent.GRU(32))
    #
    # model.add(Dense(10, activation='softmax'))

    #model2 BDLSTM
    model = Sequential()
    model.add(Bidirectional(recurrent.GRU(32,
                             activation='sigmoid',
                             inner_activation='hard_sigmoid'),
                            input_shape=(100, 2)))
    # model.add(Dropout(0.5))
    # model.add(recurrent.GRU(32))

    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(decay=1e-6)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=['accuracy'])

    X_train = np.array(X_train).reshape(-1,100,2)
    # X_test = np.array(X_test).reshape(-1,100,2)

    Y_train = to_categorical(Y_train, 10)
    # Y_test = to_categorical(Y_test, 10)

    # Fit the model
    model.fit(X_train, Y_train, nb_epoch=150, batch_size=4)

    model.save("BGRU1layerdxdy.h5")

    # model.predict(X_test, batch_size=4, verbose=0)
    # model.predict_on_batch(self, x)

    # model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    # del model  # deletes the existing model

    # returns a compiled model
    # identical to the previous one
    # model = load_model('my_model.h5')

    # scores = model.evaluate(X_test, Y_test, batch_size=4)

    # generating output result file for fusion
    # prediction = model.predict(X_test, batch_size=4)
    #
    # for i, pred in enumerate(prediction):
    #     rank = sorted(range(len(pred)),key=pred.__getitem__,
    #                   reverse=True)
    #
    #     resultString = str(np.argmax(Y_test[i])) + " " + testDataFiles[i]
    #
    #     for r in rank:
    #         resultString += " " + str(r)
    #     resultString += "\n"
    #
    #     file = open("results/" + resultFile, 'a')
    #     file.write(resultString)
    #     file.flush()
    #     os.fsync(file.fileno())
    #     file.close()

    # write accuracy results in a file
    # model.save(resultFile + ".h5")
    # scores = model.evaluate(X_test, Y_test, batch_size=4)
    # file = open("expresultFile.txt", 'a')
    # file.write("\n" + resultFile +  ":\n")
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # file.write("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # file.flush()
    # os.fsync(file.fileno())
    # file.close()


    return model

def generateModel1(X_train,Y_train,X_test,Y_test, input_length,
                   testDataFiles = "", resultFile=""):
    model = Sequential()

    # model.add(recurrent.GRU(64, input_dim=10, input_length=input_length,
    #                          activation='sigmoid',
    #                          inner_activation='hard_sigmoid'))
    #

    # BLSTM single layer

    # model.add(Bidirectional(recurrent.GRU(32,
    #                          activation='sigmoid',
    #                          inner_activation='hard_sigmoid'),
    #                         input_shape=(input_length, 10)))

    # BLSTM double layer
    model.add(Bidirectional(recurrent.GRU(32,
                                          activation='sigmoid',
                                          inner_activation='hard_sigmoid',
                            return_sequences = True),
                            input_shape=(input_length, 10)))

    model.add(recurrent.GRU(64))


    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(decay=1e-6)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=['accuracy'])

    X_train = np.array(X_train).reshape(-1,input_length,10) # datasetSize,
    # input_length, input_dimension
    # X_test = np.array(X_test).reshape(-1,input_length,10)

    Y_train = to_categorical(Y_train, 10)
    # Y_test = to_categorical(Y_test, 10)

    # Fit the model
    model.fit(X_train, Y_train, nb_epoch=150, batch_size=4)

    # model.predict(X_test, batch_size=4, verbose=0)
    # model.predict_on_batch(self, x)

    # model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    # del model  # deletes the existing model

    # returns a compiled model
    # identical to the previous one
    # model = load_model('my_model.h5')

    # scores = model.evaluate(X_test, Y_test, batch_size=4)



    # generating output result file for fusion
    # prediction = model.predict(X_test, batch_size=4)
    #
    #
    # for i, pred in enumerate(prediction):
    #     rank = sorted(range(len(pred)),key=pred.__getitem__,
    #                   reverse=True)
    #
    #     resultString = str(np.argmax(Y_test[i])) + " " + testDataFiles[i]
    #
    #     for r in rank:
    #         resultString += " " + str(r)
    #     resultString += "\n"
    #
    #     file = open("results/" + resultFile, 'a')
    #     file.write(resultString)
    #     file.flush()
    #     os.fsync(file.fileno())
    #     file.close()



    # saving the model trained on complete data
    model.save("BGRU2layerImu.h5")

    # write accuracy results in a file
    # model.save(resultFile + ".h5")
    # scores = model.evaluate(X_test, Y_test, batch_size=4)
    # file = open("expresultFile.txt", 'a')
    # file.write("\n" + resultFile +  ":\n")
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # file.write("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # file.flush()
    # os.fsync(file.fileno())
    # file.close()


    return model