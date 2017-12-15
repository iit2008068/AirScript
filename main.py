import utils.createFeature as cF
from utils import dataprep as dp
import classifiers.LSTM.model_generator as mgLSTM

import classifiers.SVM.model_generator as mgSVM

from sklearn import preprocessing

dataDir = "data/individual_all"
# trainDir = "data/check/train"
# testDir = "data/check/test"

# dxdy data
alldxdyData = "data/dxdy/allData"

# json data
alldataDir = "data/json/allData"

# trainDir = "data/json/check/train"
trainDir = "data/json/individual_all"
testDir = "data/json/check/test"
leave1outdir = "data/NewDataSplits"

leave1outTest = ["Test_Amit.txt", "Test_Arun.txt", "Test_Ayushman.txt", "Test_Pablo.txt", "Test_Roman.txt", "Test_RV.txt",
                 "Test_Shraddha.txt", "Test_Sidney.txt", "Test_Yernar.txt", "VP_Test.txt"]
leave1outTrain = ["Train_Amit.txt", "Train_Arun.txt", "Train_Ayushman.txt", "Train_Pablo.txt", "Train_Roman.txt", "Train_RV.txt",
                 "Train_Shraddha.txt", "Train_Sidney.txt", "Train_Yernar.txt", "VP_Train.txt"]

leave1outResult = ["Amit_leave1Out_new.txt","Arun_leave1Out_new.txt","Ayushman_leave1Out_new.txt","Pablo_leave1Out_new.txt","Roman_leave1Out_new.txt","RV_leave1Out_new.txt","Shraddha_leave1Out_new.txt","Sidney_leave1Out_new.txt","Yernar_leave1Out_new.txt","VP_leave1Out_new.txt"]

stratifiedTestdir = "data/startifiedDataSplitsFinal"
stratifiedTest = ["0_amit_Test.txt","1_amit_Test.txt",
                  "2_amit_Test.txt","3_amit_Test.txt",
                  "4_amit_Test.txt", "0_Arun_Test.txt",
                  "1_Arun_Test.txt","2_Arun_Test.txt",
                  "3_Arun_Test.txt","4_Arun_Test.txt",
                  "0_ayushman_Test.txt","1_ayushman_Test.txt",
                  "2_ayushman_Test.txt","3_ayushman_Test.txt",
                  "4_ayushman_Test.txt", "0_Pablo_Test.txt",
                  "1_Pablo_Test.txt","2_Pablo_Test.txt",
                  "3_Pablo_Test.txt","4_Pablo_Test.txt",
                  "0_rajveer_Test.txt","1_rajveer_Test.txt",
                  "2_rajveer_Test.txt","3_rajveer_Test.txt",
                  "4_rajveer_Test.txt","0_shraddha_Test.txt",
                  "1_shraddha_Test.txt","2_shraddha_Test.txt",
                  "3_shraddha_Test.txt","4_shraddha_Test.txt",
                  "0_VP_Test.txt","1_VP_Test.txt","2_VP_Test.txt",
                  "3_VP_Test.txt","4_VP_Test.txt",
                  "0_yernar_Test.txt","1_yernar_Test.txt",
                  "2_yernar_Test.txt","3_yernar_Test.txt",
                  "4_yernar_Test.txt","0_Roman_Test.txt",
                  "1_Roman_Test.txt","2_Roman_Test.txt",
                  "3_Roman_Test.txt","4_Roman_Test.txt",
                  "0_sidney_Test.txt","1_sidney_Test.txt",
                  "2_sidney_Test.txt","3_sidney_Test.txt",
                  "4_sidney_Test.txt"]
stratifiedTrain = ["0_amit_Train.txt","1_amit_Train.txt",
                  "2_amit_Train.txt","3_amit_Train.txt",
                  "4_amit_Train.txt", "0_Arun_Train.txt",
                  "1_Arun_Train.txt","2_Arun_Train.txt",
                  "3_Arun_Train.txt","4_Arun_Train.txt",
                  "0_ayushman_Train.txt","1_ayushman_Train.txt",
                  "2_ayushman_Train.txt","3_ayushman_Train.txt",
                  "4_ayushman_Train.txt", "0_Pablo_Train.txt",
                  "1_Pablo_Train.txt","2_Pablo_Train.txt",
                  "3_Pablo_Train.txt","4_Pablo_Train.txt",
                  "0_rajveer_Train.txt","1_rajveer_Train.txt",
                  "2_rajveer_Train.txt","3_rajveer_Train.txt",
                  "4_rajveer_Train.txt","0_shraddha_Train.txt",
                  "1_shraddha_Train.txt","2_shraddha_Train.txt",
                  "3_shraddha_Train.txt","4_shraddha_Train.txt",
                  "0_VP_Train.txt","1_VP_Train.txt","2_VP_Train.txt",
                  "3_VP_Train.txt","4_VP_Train.txt",
                  "0_yernar_Train.txt","1_yernar_Train.txt",
                  "2_yernar_Train.txt","3_yernar_Train.txt",
                  "4_yernar_Train.txt","0_Roman_Train.txt",
                  "1_Roman_Train.txt","2_Roman_Train.txt",
                  "3_Roman_Train.txt","4_Roman_Train.txt",
                  "0_sidney_Train.txt","1_sidney_Train.txt",
                  "2_sidney_Train.txt","3_sidney_Train.txt",
                  "4_sidney_Train.txt"]
stratifiedResult = ["0_amit_Result.txt","1_amit_Result.txt",
                  "2_amit_Result.txt","3_amit_Result.txt",
                  "4_amit_Result.txt", "0_Arun_Result.txt",
                  "1_Arun_Result.txt","2_Arun_Result.txt",
                  "3_Arun_Result.txt","4_Arun_Result.txt",
                  "0_ayushman_Result.txt","1_ayushman_Result.txt",
                  "2_ayushman_Result.txt","3_ayushman_Result.txt",
                  "4_ayushman_Result.txt", "0_Pablo_Result.txt",
                  "1_Pablo_Result.txt","2_Pablo_Result.txt",
                  "3_Pablo_Result.txt","4_Pablo_Result.txt",
                  "0_rajveer_Result.txt","1_rajveer_Result.txt",
                  "2_rajveer_Result.txt","3_rajveer_Result.txt",
                  "4_rajveer_Result.txt","0_shraddha_Result.txt",
                  "1_shraddha_Result.txt","2_shraddha_Result.txt",
                  "3_shraddha_Result.txt","4_shraddha_Result.txt",
                  "0_VP_Result.txt","1_VP_Result.txt",
                    "2_VP_Result.txt",
                  "3_VP_Result.txt","4_VP_Result.txt",
                  "0_yernar_Result.txt","1_yernar_Result.txt",
                  "2_yernar_Result.txt","3_yernar_Result.txt",
                  "4_yernar_Result.txt","0_Roman_Result.txt",
                  "1_Roman_Result.txt","2_Roman_Result.txt",
                  "3_Roman_Result.txt","4_Roman_Result.txt",
                  "0_sidney_Result.txt","1_sidney_Result.txt",
                  "2_sidney_Result.txt","3_sidney_Result.txt",
                  "4_sidney_Result.txt"]

if __name__ == '__main__':
    # points = dp.loadPoints()
    # cF.angle(points)

    # dxdy on all data
    data_x, data_y = dp.loadData(dataDir)
    model = mgLSTM.generateModel(data_x, data_y, 0, 0)

    '''
    # stratified k fold
    data_x, data_y = dp.loadData(dataDir)
    nFolds = dp.createFolds(data_x,data_y,5)
    for i in range(0,len(nFolds)):
        if i is 3:
            train = nFolds[i][0]
            validation = nFolds[i][1]
            model = mgLSTM.generateModel(train[0],train[1],validation[0],validation[1])
            # print "Fold: " + str(i) + " Accuracy score: " + str(acc_scr) + "\n\n"

    '''


    '''
    # leave 1 out test for dxdy data

    for i in range(0, len(leave1outTest)):
        train_x, train_y, _ = dp.loadDataFromList(leave1outdir +
                                                      "/"
                                                      + leave1outTrain[
                                                          i], alldxdyData)
        test_x, test_y, testDataFiles = dp.loadDataFromList(
            leave1outdir +
                                                  "/"
                                                  + leave1outTest[i], alldxdyData)

        model = mgLSTM.generateModel(train_x, train_y,test_x,
                                     test_y, testDataFiles,
                                      "dxdy_" + leave1outResult[i])
                # print "Fold: " + str(i) + " Accuracy score: " + str(acc_scr) + "\n\n"

    '''
    '''
    # Stratified test, dxdy data
    for i in range(0, len(stratifiedTest)):
        train_x, train_y, _ = dp.loadDataFromList(stratifiedTestdir +
                                                      "/"
                                                      + stratifiedTrain[
                                                          i], alldxdyData)
        test_x, test_y, testDataFiles = dp.loadDataFromList(
            stratifiedTestdir +
                                                  "/"
                                                  + stratifiedTest[i], alldxdyData)

        model = mgLSTM.generateModel(train_x, train_y,test_x,
                                     test_y, testDataFiles,
                                      "dxdy_" + stratifiedResult[i])
                # print "Fold: " + str(i) + " Accuracy score: " + str(acc_scr) + "\n\n"
    '''

    '''
    # LSTM using IMU data
    scaler = preprocessing.MaxAbsScaler()
    _, train_x, train_y, _, _, _, _, _, _, _, _, \
    _ = dp.getDataJson(trainDir, scaler)

    train_x, maxTrainSteps = dp.addPadding(train_x)

    _, test_x, test_y, _, _, _, _, _, _, _, _, _ = dp.getDataJson(
        testDir, scaler)

    test_x, _ = dp.addPadding(test_x, maxTrainSteps)

    model = mgLSTM.generateModel1(train_x, train_y, test_x,
                                 test_y, maxTrainSteps)
    # print "Fold: " + str(i) + " Accuracy score: " + str(acc_scr)
    # + "\n\n"

    '''
    '''
    # leave 1 out tests
    scaler = preprocessing.MaxAbsScaler()
    for i in range(0, len(leave1outTest)):
        if i is not 3:
            continue
        train_x, train_y, _ = dp.getDataJsonFromList(leave1outdir +
                                                  "/"
                                                  + leave1outTrain[
                                                      i], scaler, alldataDir)

        train_x, maxTrainSteps = dp.addPadding(train_x)
        test_x, test_y, testDataFiles = dp.getDataJsonFromList(
            leave1outdir +
                                                  "/"
                                                  + leave1outTest[i],
                                                scaler, alldataDir)
        test_x, _ = dp.addPadding(test_x, maxTrainSteps)

        model = mgLSTM.generateModel1(train_x, train_y, test_x,
                                 test_y, maxTrainSteps,
                                      testDataFiles,
                                      leave1outResult[i])
    '''

    '''
    # Stratified test, imu data
    scaler = preprocessing.MaxAbsScaler()
    for i in range(0, len(stratifiedTest)):
        train_x, train_y, _ = dp.getDataJsonFromList(
            stratifiedTestdir +
            "/"
            + stratifiedTrain[
                i], scaler, alldataDir)

        train_x, maxTrainSteps = dp.addPadding(train_x)
        test_x, test_y, testDataFiles = dp.getDataJsonFromList(
            stratifiedTestdir +
            "/"
            + stratifiedTest[i],
            scaler, alldataDir)
        test_x, _ = dp.addPadding(test_x, maxTrainSteps)

        model = mgLSTM.generateModel1(train_x, train_y, test_x,
                                      test_y, maxTrainSteps,
                                      testDataFiles,
                                      stratifiedResult[i])
    '''


# parameters to do a grid search on SVM
svm_tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6],
                     'C': [1, 10, 100, 1000, 10000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]}]

dataDir = "data/individual_all"

def SvmClassification() :
    # points = dp.loadPoints()
    # cF.angle(points)
    data_x, data_y = dp.loadData(dataDir)
    nFolds = dp.createFolds(data_x,data_y,5)
    for i in range(0,len(nFolds)):
        train = nFolds[i][0]
        validation = nFolds[i][1]
        model, clf_rpt, cm, acc_scr, best_params = mgSVM.generateModel(train[0],train[1],validation[0],validation[1], svm_tuned_parameters)
        print("Fold: " + str(i) + " Accuracy score: " + str(
            acc_scr) + "\n\n")