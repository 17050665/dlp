import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import csv

def draw_scatter(x, y, colorStr):
    plt.scatter(x, y, s=20, color=colorStr)

def plot(W, b, filename):
    x01 = []
    x02 = []
    x11 = []
    x12 = []
    ds_reader = csv.reader(open(filename, encoding='utf-8'))
    for row in ds_reader:
        if int(row[0]) > 0:
            x11.append(float(row[1]))
            x12.append(float(row[2]))
        else:
            x01.append(float(row[1]))
            x02.append(float(row[2]))
    draw_scatter(x01, x02, 'r')
    draw_scatter(x11, x12, 'b')
    
    w1 = W[0][0]
    w2 = W[1][0]
    b_ = b[0]
    x1 = np.linspace(-0.2, 1.2, 100)
    x2 = [-(w1/w2)*x-b_/w2 for x in x1]
    plt.plot(x1, x2, 'g-', label='Plan', linewidth=2)
    
    plt.show()
    