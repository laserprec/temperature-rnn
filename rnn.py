import csv
import glob
import math
import time as t
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

DEBUG = True
DATETIME_MASK = '%d.%m.%Y %H:%M:%S'

def parseDate(dateStr):
    """
    Replaces datetime.strptime() for performance gain
    """
    # if DEBUG: print('dateStr[6:9]:', dateStr[6:10], '| dateStr[3:5]:', dateStr[3:5], '| dateStr[:2]:', dateStr[:3], \
    #                 '| dateStr[11:13]:', dateStr[11:13], '| dateStr[14:16]:', dateStr[14:16], '| dateStr[17:19]', dateStr[17:19])
    return datetime(
                int(dateStr[6:10]), #year
                int(dateStr[3:5]),  #month
                int(dateStr[:2]),   #day
                int(dateStr[11:13]), #hour
                int(dateStr[14:16]), #minute
                int(dateStr[17:19]), #second
                )

def readFile(filename):
    with open(filename, newline='') as csvfile:
        row_cnt = len(csvfile.readlines()) - 1 # not counting the header row
        csvfile.seek(0) # reset file ptr to start of file
        reader = csv.reader(csvfile)
        next(reader, None) # skip reading the header row

        time = np.zeros(row_cnt, dtype=datetime)
        temp = np.zeros(row_cnt, dtype=float)
        index = 0
        for row in reader:
            temperate = np.float(row[2])
            date = parseDate(row[0])
            if -100 < temperate < 100: # ignore error measurements
                time[index] = date
                temp[index] = temperate
                index += 1
            # else:
                # if DEBUG: print(f"outliner: {temperate}, time: {row[0]}")
                
    # truncate the empty cells in the array
    return time[:index], temp[:index]

def readData():
    filenames = glob.glob('./data/*.csv')
    data = {"time": np.array([]), "temp": np.array([])}
    for fn in filenames:
        start = t.time()
        time, temp = readFile(fn)
        end = t.time()
        data["time"] = np.append(data["time"], time)
        data["temp"] = np.append(data["temp"], temp)
        if DEBUG: print(f"Total duration for reading {len(temp)}: {end - start:.3f}")
    return data

def plotGraph(date, data):
    years = mdates.YearLocator()   # every year
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
    data = readData()
    start = t.time()
    plt.plot(data["temp"])
    # plotGraph(data["time"], data["temp"])
    plt.draw()
    end = t.time()
    if DEBUG: print(f"Total duration for plotting: {end - start:.3f}")
    plt.show()
if __name__ == '__main__':
    main()