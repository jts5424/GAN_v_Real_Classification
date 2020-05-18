# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:53:31 2020

@author: JTSDellLaptop
"""

import numpy as np
import matplotlib.pyplot as plt

# Plot two sets of data on same figure
def dual_plot(x,data1,data2,label1,label2,xlabel,ylabel,title,filename):

    fig, ax = plt.subplots()
    ax.plot(x,data1,label=label1)
    ax.plot(x,data2,label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(filename)
