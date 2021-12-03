import numpy as np
import json

count = 1
punch_data = []
with open("./attack_1203.txt", "r") as f:
    for data in f:
        data = json.loads(data)
        punch_data.append(data)
    #punch_data = np.array(punch_data)
    print(np.shape(punch_data))

defense_data = []
with open("./defense_1203.txt", "r") as f:
    for data in f:
        data = json.loads(data)
        defense_data.append(data)
    #defense_data = np.array(defense_data)
    print(np.shape(defense_data))

skill1_data = []
with open("./skill1_1203.txt", "r") as f:
    for data in f:
        data = json.loads(data)
        skill1_data.append(data)
    #skill1_data = np.array(skill1_data)
    print(np.shape(skill1_data))

wait_data = []
with open("./wait_1203.txt", "r") as f:
    for data in f:
        data = json.loads(data)
        wait_data.append(data)
    print(np.shape(wait_data))

with open("./x_train.txt", "a") as fi:
    for i in punch_data:
        fi.write((str(i) + "\n"))

with open("./x_train.txt", "a") as fi:
    for i in defense_data:
        fi.write((str(i) + "\n"))

with open("./x_train.txt", "a") as fi:
    for i in skill1_data:
        fi.write((str(i) + "\n"))

with open("./x_train.txt", "a") as fi:
    for i in wait_data:
        fi.write((str(i) + "\n"))

with open("./y_train.txt", "a") as fi:
    for i in range(0, 1000):
        fi.write(str([1, 0, 0, 0]) + "\n")

    for i in range(0, 1000):
        fi.write(str([0, 1, 0, 0]) + "\n")

    for i in range(0, 1000):
        fi.write(str([0.1, 0.1, 0.7, 0.1]) + "\n")

    for i in range(0, 1000):
        fi.write(str([0, 0, 0, 1]) + "\n")