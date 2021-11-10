import os
import numpy as np

path = "./test_data/all/"
allFileList = os.listdir(path)
print(allFileList)

frame = []
count = 1
for file in allFileList:
    if count > 50:
        count = 1
        frame = []
    with open("./test_data/all/" + file, 'r') as f:
        arr = []
        #frame = []
        #print(type(file))
        for data in f:
            data = data.rstrip("\n")
            data = data.split(", ")
            data = list(map(float, data))
            data = np.array(data)[1:]
            data = list(data)
            arr.append(data)
            if len(arr) == 13:
                temp = list(np.array(arr).reshape(39))
                arr = []
                frame.append(temp)
            if len(frame) == 20:
                print(np.shape(frame))
                with open("./test_data/x_train1.txt", "a") as fi:
                    fi.write(str(frame) + "\n")
                if "punch" in file:
                    y_train = [1, 0, 0, 0]
                if "Defense" in file:
                    y_train = [0, 1, 0, 0]
                if "skillone" in file:
                    y_train = [0, 0, 1, 0]
                if "skilltwo" in file:
                    y_train = [0, 0, 0, 1]
                with open("./test_data/y_train1.txt", "a") as fi:
                    fi.write(str(y_train) + "\n")
                frame = frame[1:]

    count += 1