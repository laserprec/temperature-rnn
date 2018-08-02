import csv
import glob
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

def readFile(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip reading the header row
        data = list(reader)
        row_cnt = len(data)
        datetime = np.zeros(row_cnt, dtype=str)
        temp = np.zeros(row_cnt, dtype=float)
        for index in range(row_cnt):
            temperate = np.float(data[index][2])
            if -100 < temperate < 100:
                datetime[index] = data[index][0]
                temp[index] = temperate
            else:
                print(f"outliner: {temperate}, timer: {data[index][0]}")
    return datetime, temp

def readData():
    filenames = glob.glob('./data/*.csv')
    data = {"datetime": np.array([]), "temp": np.array([])}
    for fn in filenames:
        start = time.time()
        datetime, temp = readFile(fn)
        end = time.time()
        data["datetime"] = np.append(data["datetime"], datetime)
        data["temp"] = np.append(data["temp"], temp)
        print(f"Total duration for reading {len(data['temp'])}: {end - start:.3f}")
    return data

def main():
    data = readData()
    d = data["temp"][:300000]
    start = time.time()
    plt.plot(d)
    plt.draw()
    end = time.time()
    print(f"Total duration for plotting: {end - start:.3f}")
    plt.show()
if __name__ == '__main__':
    main()