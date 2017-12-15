import numpy as np
import os
import json
import utils.createFeature as cF
import math as mt
from scipy.interpolate import interp1d
from scipy.signal import resample
from sklearn.cross_validation import StratifiedKFold

def remove_redundantdata(points, threshold = 5):
    index = []

    j = 1
    while j < len(points):
        if mt.sqrt(mt.pow(points[j][0] - points[j-1][0],2) + mt.pow(points[j][1] - points[j-1][1],2)) < 5:
            index.append(j)
            j += 2
        else:
            j += 1
    points = np.delete(points, index, axis = 0)

    # index = []
    # apply again for initial points and ending points
    # for j in range(1, len(points)):
    #     if mt.sqrt(mt.pow(points[j][0] - points[j - 1][0], 2) + mt.pow(points[j][1] - points[j - 1][1], 2)) < 5:
    #         index.append(j)
    #
    # points = np.delete(points, index, axis=0)
    if len(points) < 10:
        print(points)
    # np.savetxt("nonredundant_points.txt",points, delimiter=' ')
    return points

def interpolatedata(points, size = 100):
    x = np.asarray(range(0, len(points)),dtype=float)
    # normalize x
    x = x/(len(x)-1)
    f2 = interp1d(x, points, kind='linear', axis=0)
    m = np.asarray(range(0, size), dtype=float)
    m = m/(len(m)-1)
    points = f2(m)
    return points

def smoothdata(points, n = 2):

    for j in range(n,len(points)- n):
        pointx = points[j][0]
        pointy = points[j][1]
        for k in range(1,n+1):
            pointx += points[j-k][0] + points[j+k][0]
            pointy += points[j-k][1] + points[j+k][1]
        pointx = pointx/(2*n+1)
        pointy = pointy/(2*n+1)
        points[j] = [pointx,pointy]
    # np.savetxt("aftersmooth_points.txt", points, delimiter=' ')
    return points

def normalizedata(points):
    '''
    normalize in -1 to 1 box
    :param points:
    :return:
    '''
    max = np.amax(points, axis=0)
    min = np.amin(points, axis=0)

    for j in range(0,len(points)):
        pointx = (points[j][0] - min[0])/(max[0] - min[0])
        pointy = (points[j][1] - min[1])/(max[1] - min[1])
        pointx = 2*pointx - 1
        pointy = 2*pointy - 1

        points[j] = [pointx,pointy]
    # np.savetxt("afternormalize_points.txt", points, delimiter=' ')
    return points

def normalizedata1(points):
    '''
    normalize by subtracting mean and dividing by std
    :param points:
    :return:
    '''
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)

    for j in range(0,len(points)):
        pointx = (points[j][0] - mean[0])/(std[0])
        pointy = (points[j][1] - mean[1])/(std[1])

        points[j] = [pointx,pointy]
    # np.savetxt("afternormalize_points.txt", points, delimiter=' ')
    return points

def loadPoints(fname="data/0_0_amit.txt"):
    points = np.loadtxt(fname, dtype='string')
    pointsx = points[:,1]
    pointsy = points[:,3]

    pointsx = np.array(pointsx).astype(float)

    pointsy = np.array(pointsy).astype(float)

    points = zip(pointsx, pointsy)

    for i,point in enumerate(points):
        points[i] = np.array(point)
    # np.savetxt(
    #     "/media/amit/Windows8_OS/Users/Amit/Documents/AirScript"
    #     "/preprocessed_data/allPoints/"+file,
    #     points, delimiter=' ')

    points = smoothdata(points)
    points = remove_redundantdata(points)
    points = normalizedata1(points)
    points = interpolatedata(points)
    # np.savetxt("C:/Users/Amit/Documents/AirScript/preprocessed_data/afterpreprocess_points.txt", points, delimiter=' ')
    return points

def loadData(rootdir):
    training_class_dirs = os.walk(rootdir)
    labeldirs = []
    labels = []
    skip = True
    data_x = []
    data_y = []

    for trclass in training_class_dirs:
        #print(trclass)
        if skip is True:
            labels = trclass[1]
            skip = False
            continue
        labeldirs.append((trclass[0],trclass[2]))

    j = -1
    for i,labeldir in enumerate(labeldirs):
        saveDirPath = ""
        dirPath = labeldir[0]
        filelist = labeldir[1]
        if not bool(filelist):
            j += 1
            continue

        for file in filelist:
            fname = os.path.join(dirPath, file)
            points = loadPoints(fname)    # load point data
            # feat = cF.angle(points)
            # feat = cF.direction(points)
            feat = points

            data_x.append(feat)
            data_y.append(labels[j])

    return data_x, data_y

def loadDataFromList(splitfilename, dataDir):
    data_x = []
    data_y = []

    DataFiles = []

    with open(splitfilename, 'r') as f:
        for line in f:
            fsplit = line.split(" ")

            tar_val = int(fsplit[1])
            split = fsplit[0].split("/")
            file = split[2]

            DataFiles.append(file)

            file = file.split(".")[0] + ".txt"
            fname = os.path.join(dataDir, file)
            points = loadPoints(fname)  # load point data
            # feat = cF.angle(points)
            # feat = cF.direction(points)
            feat = points

            data_x.append(feat)
            data_y.append(tar_val)

    return data_x, data_y, DataFiles


def splitDataset(train,test,target,data):
    train_x = []
    train_y = []

    for i in train:
        train_x.append(data[i])
        train_y.append(target[i])

    val_x = []
    val_y = []

    for i in test:
        val_x.append(data[i])
        val_y.append(target[i])

    return train_x, train_y, val_x, val_y

def createFolds(data_x, data_y, n_folds = 5):
    skf = StratifiedKFold(data_y, n_folds, shuffle=True)
    kFolds = []

    for train, test in skf:
        train_x, train_y, val_x, val_y = splitDataset(train, test, data_y, data_x)
        train = (train_x, train_y)
        validation = (val_x, val_y)
        kFolds.append((train, validation))
    return kFolds

def oneHotRepresentation(labels):
    b = np.zeros((np.array(labels).size, 10))
    b[np.arange(np.array(labels).size), labels] = 1
    return b

def loadSignalData(rootdir):
    training_class_dirs = os.walk(rootdir)
    labeldirs = []
    skip = True

    for trclass in training_class_dirs:
        #print(trclass)
        if skip is True:
            skip = False
            continue
        labeldirs.append((trclass[0],trclass[2]))


    for i,labeldir in enumerate(labeldirs):
        saveDirPath = ""
        dirPath = labeldir[0]
        filelist = labeldir[1]
        if not bool(filelist):
            # extract synthetic data save location
            saveDirPath = dirPath
            continue

        # synthetic data for the same person, e.g. signal8_1 and signal8_2 are combined
        for file1 in filelist:
            f1 = []
            y1len = []

            fileData1 = read_json_file(dirPath + os.path.sep + file1)

            for index, y1 in enumerate([fileData1['emg']['data'], fileData1['acc']['data'],
                                        fileData1['gyr']['data'],
                                        fileData1['ori']['data']]):
                x1 = np.asarray(range(0, len(y1)))
                x1 = x1 / (len(x1) - 1)
                f1.append(interp1d(x1, y1, kind='cubic', axis=0))
                y1len.append(len(y1))

            for file2 in filelist:
                if file1 >= file2:
                    continue

                fileData2 = read_json_file(dirPath + os.path.sep + file2)
                newfiledata = []
                newSignaldata = []
                for index, y2 in enumerate(
                        [fileData2['emg']['data'], fileData2['acc']['data'], fileData2['gyr']['data'],
                         fileData2['ori']['data']]):
                    x2 = np.asarray(range(0, len(y2)))
                    x2 = x2 / (len(x2) - 1)
                    f2 = interp1d(x2, y2, kind='cubic', axis=0)

                    mlen = int((y1len[index] + len(y2)) / 2)
                    m = np.asarray(range(0,mlen))
                    m = m/(len(m)-1)
                    nsignal1 = f1[index](m)
                    nsignal2 = f2(m)
                    newSignaldata.append((np.array((nsignal1 + nsignal2)/2)).tolist())
                newfiledata = {'emg':{'timestamps':fileData2['emg']['timestamps'], 'data':newSignaldata[0]}, 'acc':{'timestamps':fileData2['acc']['timestamps'], 'data' : newSignaldata[1]}, 'gyr':{'timestamps':fileData2['gyr']['timestamps'], 'data':newSignaldata[2]}, 'ori':{'timestamps':fileData2['ori']['timestamps'], 'data':newSignaldata[3]} }


def read_json_file(filepath):
    with open(filepath) as data_file:
        data = json.load(data_file)
        return data

def getDataJson(rootdir, scaler):
    '''
    This method gets all the training data from the root directory of the ground truth
    The following is the directory structure for the ground truth
    Root_Dir
        |_Labels
            |_Participants
                |_data_files
    :param rootdir (string): path to the rood directory where the ground truth exists
    :return:    labels      (list),                 A list of class labels
                data        (list),                 A list of training instances
                target      (list),                 A list of class labels corresponding to the training instances in the in 'data'
                labelsdict  (dictionary),           A dictionary for converting class labels from string to integer and vice versa
                avg_len     (float),                The average length of the sensor data (emg, accelerometer, gyroscope and orientation) which would later be used for normalization
                user_map    (dictionary),           A dictionary of all participants and their corresponding file list to be used for leave one out test later
                user_list   (list),                 A list of all participants
                data_dict   (dictionary)            A dictionary containing a mapping of all the class labels, participants, and files of the participants which can be used later for transforming the data for leave one out test
                max_len     (integer)               the maximum length of the sensor data
                data_path   (list)                  A list that will hold the path to every training instance in the 'data list'
    '''

    # List of all training labels
    training_class_dirs = os.walk(rootdir)


    labels = []                         # Empty list to hold all the class labels
    labelsdict = {}                     # Dictionary to store the labels and their correspondig interger values
    labeldirs = []                      # Directory paths of all labels
    target = []                         # List that will hold class labels of the training instances in 'data list'
    data = []                           # List that will hold all the training/validation instances
    sample_len_vec_emg = []             # List that holds that length of all the sensor data. It will be used later for calculating average length
    sample_len_vec_others = []          # List that holds that length of all the sensor data. It will be used later for calculating average length
    data_dict = {}                      # The dictionary that will hold that mappings for all labels, participants of the the label and data files corresponding to all the participants. This will be used later for leave one out test
    user_map = {}                       # A dictionary that will hold the mappings for all participants and their corresponding ids
    user_list = []                      # A list of all participants
    user_ids = np.arange(100).tolist()  # A pre generated list of userids for assigning a unique id to every user
    data_path = []                      # A list that will hold the path to every training instance in the 'data list'

    # Get the list of labels by walking the root directory
    for trclass in training_class_dirs:
        labels = trclass[1]
        break

    # extracting list of participants for each label
    for i,label in enumerate(labels):
        dict = {}                                   # dictionary to store participant information
        lbl_users_lst = []                          # list of participants per label
        labelsdict[label] = i
        labeldir = os.path.join(rootdir,label)

        #list of users for the respective label
        lbl_usrs = os.walk(labeldir)

        #enumerating all the users of the respective label
        for usr in lbl_usrs:
            #print(usr)
            lbl_users_lst = usr[1]

            #assigning unique ids to all the users
            for i,user in enumerate(lbl_users_lst):
                if user not in user_map:
                    id = user_ids.pop()
                    user_map[user] =id
                    user_list.append(id)
            break

        #extracting data file list for every  participant
        for usr in lbl_users_lst:
            usrdir = os.path.join(labeldir,usr)
            filewalk = os.walk(usrdir)
            file_list = []
            for fls in filewalk:
                file_list = fls[2]
                break
            dict[usr] = (usrdir,file_list)

        dict['users'] = lbl_users_lst
        data_dict[label] = dict                 # add all meta information to data_dict

    # Extracting data from the data files from all participants
    for key, value in data_dict.items():
        tar_val = int(key)
        users = value['users']
        for user in users:
            user_dir = value[user]
            dirPath = user_dir[0]
            filelist = user_dir[1]
            for file in filelist:
                fp = os.path.join(dirPath,file)

                data_path.append(fp)

                fileData = read_json_file(fp)
                # extract data from the dictionary
                # emg
                emg = np.array(fileData['emg']['data'])
                emgts = np.array(fileData['emg']['timestamps'])

                # accelerometer
                acc = np.array(fileData['acc']['data'])
                accts = np.array(fileData['acc']['timestamps'])

                # gyroscope
                gyr = np.array(fileData['gyr']['data'])
                gyrts = np.array(fileData['gyr']['timestamps'])

                # orientation
                ori = np.array(fileData['ori']['data'])
                orits = np.array(fileData['ori']['timestamps'])

                # create training instance
                accList, gyrList, oriList = separateRawData(acc,gyr,ori)

                #scaling data
                accList, gyrList, oriList = scaledatactc(accList, gyrList, oriList, scaler)

                ti = np.concatenate((accList, gyrList, oriList), axis=0)  # consolidated data
                #ti = tri.TrainingInstance(key, emg, acc, gyr, ori, emgts, accts, gyrts, orits)

                # add length for resampling later to the sample length vector
                sample_len_vec_emg.append(emg.shape[0])
                sample_len_vec_others.append(acc.shape[0])

                # append training instance to data list
                data.append(ti)

                # append class label to target list
                target.append(tar_val)
    avg_len_emg = int(np.mean(sample_len_vec_emg))
    avg_len_acc = int(np.mean(sample_len_vec_others))
    max_length_emg = np.amax(sample_len_vec_emg)
    max_length_others = np.amax(sample_len_vec_others)
    return labels,data,target,labelsdict,avg_len_emg,avg_len_acc,user_map,user_list,data_dict,max_length_emg,max_length_others,data_path

def separateRawData(acc, gyr, ori):
    if acc is not None:
        accList = np.array([np.array(acc)[:, 0], np.array(acc)[:, 1], np.array(acc)[:, 2]])

    if gyr is not None:
        gyrList = np.array([np.array(gyr)[:, 0], np.array(gyr)[:, 1], np.array(gyr)[:, 2]])

    if ori is not None:
        oriList = np.array(
            [np.array(ori)[:, 0], np.array(ori)[:, 1], np.array(ori)[:, 2], np.array(ori)[:, 3]])

    return accList, gyrList, oriList

def scaledatactc(accList, gyrList, oriList, scaler):
    # scaling data
    norm_accs = []
    norm_gyrs = []
    norm_oris = []
    for a, b in zip(accList, gyrList):
        a = a.reshape(-1, 1)
        a = scaler.fit_transform(a)
        reshaped_a = a.reshape(a.shape[0])
        norm_accs.append(reshaped_a)
        b = b.reshape(-1, 1)
        b = scaler.fit_transform(b)
        reshaped_b = b.reshape(a.shape[0])
        norm_gyrs.append(reshaped_b)

    for x in oriList:
        x = x.reshape(-1, 1)
        x = scaler.fit_transform(x)
        reshaped = x.reshape(x.shape[0])
        norm_oris.append(reshaped)
    return np.array(norm_accs), np.array(norm_gyrs), np.array(norm_oris)

def addPadding(inputlist, trainMaxSteps=0):

    resampledList = []
    maxSteps = 0
    for inp in inputlist:
        maxSteps = max(maxSteps, inp.shape[1])

    if (trainMaxSteps):
        maxSteps = trainMaxSteps

    for i in range(0, len(inputlist)):
        # padSecs = maxSteps - inputlist[i].shape[1]
        # if (padSecs >= 0):
        #     inputlist[i] = np.pad(inputlist[i].T,
        #                                        ((0, padSecs), (0, 0)),
        #                                        'constant',
        #                                        constant_values=0)
        # else:
        #     inputlist[i] = resample(inputlist[i].T, maxSteps,
        #                   axis=0)
        resampledList.append(resample(inputlist[i].T, maxSteps,
                           axis=0))

    return resampledList, maxSteps

def getDataJsonFromList(splitfilename, scaler, dataDir):

    data = []
    target = []

    DataFiles = []

    with open(splitfilename, 'r') as f:
        for line in f:
            fsplit = line.split(" ")

            tar_val = int(fsplit[1])
            split = fsplit[0].split("/")
            file = split[2]
            DataFiles.append(file)

            fp = os.path.join(dataDir, file)

            fileData = read_json_file(fp)
            # extract data from the dictionary
            # emg
            emg = np.array(fileData['emg']['data'])
            emgts = np.array(fileData['emg']['timestamps'])

            # accelerometer
            acc = np.array(fileData['acc']['data'])
            accts = np.array(fileData['acc']['timestamps'])

            # gyroscope
            gyr = np.array(fileData['gyr']['data'])
            gyrts = np.array(fileData['gyr']['timestamps'])

            # orientation
            ori = np.array(fileData['ori']['data'])
            orits = np.array(fileData['ori']['timestamps'])

            # create training instance
            accList, gyrList, oriList = separateRawData(acc, gyr, ori)

            # scaling data
            accList, gyrList, oriList = scaledatactc(accList, gyrList,
                                                     oriList, scaler)

            ti = np.concatenate((accList, gyrList, oriList),
                                axis=0)  # consolidated data
            # ti = tri.TrainingInstance(key, emg, acc, gyr, ori,
            # emgts, accts, gyrts, orits)

            # append training instance to data list
            data.append(ti)

            # append class label to target list
            target.append(tar_val)

    return data, target, DataFiles