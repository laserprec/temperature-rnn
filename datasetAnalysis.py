import csv
import glob
import pickle
import time as t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

DEBUG = True
DATETIME_MASK = '%d.%m.%Y %H:%M:%S'

TRAINING_SET = './data/train/*.csv'
VALIDATION_SET = './data/validate/*.csv'
TEST_SET = './data/test/*.csv'

TRAINING_SET_STAT = './data/train/stat.pickle'
VALIDATION_SET_STAT = './data/validate/stat.pickle'
TEST_SET_STAT = './data/test/stat.pickle'

def parseDate(dateStr):
    """
    Replaces datetime.strptime() for performance gain
    """
    # if DEBUG: print('dateStr[6:9]:', dateStr[6:10], '| dateStr[3:5]:', dateStr[3:5], '| dateStr[:2]:', dateStr[:3], \
    #                 '| dateStr[11:13]:', dateStr[11:13], '| dateStr[14:16]:', dateStr[14:16], '| dateStr[17:19]', dateStr[17:19])
    return datetime(
                int(dateStr[6:10]),  #year
                int(dateStr[3:5]),   #month
                int(dateStr[:2]),    #day
                int(dateStr[11:13]), #hour
                int(dateStr[14:16]), #minute
                int(dateStr[17:19]), #second
                )

def readFile(filename):
    with open(filename, newline='') as csvfile:
        row_cnt = len(csvfile.readlines()) - 1  # not counting the header row
        csvfile.seek(0)                         # reset file ptr to start of file
        reader = csv.reader(csvfile)
        next(reader, None)                      # skip reading the header row

        time = np.zeros(row_cnt, dtype=datetime)
        temp = np.zeros(row_cnt, dtype=float)
        index = 0
        for row in reader:
            temperate = np.float(row[2])
            date = parseDate(row[0])
            if -100 < temperate < 100:          # ignore error measurements
                time[index] = date
                temp[index] = temperate
                index += 1
            # else:
                # if DEBUG: print(f"outliner: {temperate}, time: {row[0]}")
                
    # truncate the empty cells in the array
    return time[:index], temp[:index]

def readDataset(filenames):
    # data = {"time": np.array([]), "temp": np.array([])}
    data = np.array([])
    for fn in filenames:
        start = t.time()
        time, temp = readFile(fn)
        end = t.time()
        # data["time"] = np.append(data["time"], time)
        # data["temp"] = np.append(data["temp"], temp)
        data = np.append(data, temp)
        if DEBUG: print(f"Total duration for reading {len(temp)}: {end - start:.3f}")
    return data

def getStatistic(data):
    return np.std(data), np.mean(data), data.shape

def plotGraph(date, data):
    years = mdates.YearLocator()    # every year
    months = mdates.MonthLocator()  # every month
    dates = mdates.DateLocator()
    monthFmt = mdates.DateFormatter('%m-%Y')
    yearsFmt = mdates.DateFormatter('%M')

    # datemin = np.datetime64(date[0], 'Y')
    # datemax = np.datetime64(date[-1], 'Y') + np.timedelta64(1, 'Y')

    fig, ax = plt.subplots()
    ax.plot(date, data)

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_major_formatter(monthFmt)

    # ax.set_xlim(datemin, datemax)

    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.autofmt_xdate()

def main():
    train = readDataset(glob.glob(TRAINING_SET))
    validate = readDataset(glob.glob(VALIDATION_SET))
    test = readDataset(glob.glob(TEST_SET))
    
    std, mean, shape = getStatistic(train)
    with open(TRAINING_SET_STAT, 'wb') as f: pickle.dump({"std": std, "mean": mean, "shape": shape}, f)
    print(f"Training set - std: {std}, mean: {mean}, shape: {shape}")
    print(f"Storing the mean and std of training set in {TRAINING_SET_STAT}")

    std, mean, shape = getStatistic(validate)
    with open(VALIDATION_SET_STAT, 'wb') as f: pickle.dump({"std": std, "mean": mean, "shape": shape}, f)
    print(f"Validation set - std: {std}, mean: {mean}, shape: {shape}")
    print(f"Storing the mean and std of validation set in {VALIDATION_SET_STAT}")

    std, mean, shape = getStatistic(test)
    with open(TEST_SET_STAT, 'wb') as f: pickle.dump({"std": std, "mean": mean, "shape": shape}, f)
    print(f"Test set - std: {std}, mean: {mean}, shape: {shape}")
    print(f"Storing the mean and std of validation set in {TEST_SET_STAT}")

    start = t.time()
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 8))
    ax1.plot(train)
    ax1.set_title('Training Set')
    ax2.plot(validate)
    ax2.set_title('Validation Set')
    ax3.plot(test)
    ax3.set_title('Test Set')
    plt.subplots_adjust(hspace=.4)
    # plotGraph(data["time"], data["temp"])
    plt.draw()
    end = t.time()

    if DEBUG: print(f"Total duration for plotting: {end - start:.3f}")
    plt.show()
if __name__ == '__main__':
    main()