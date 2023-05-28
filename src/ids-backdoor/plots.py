import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import Counter
import textwrap



def plot_hist(output, classes):
    labels, counts = np.unique(output,return_counts=True)
    ticks = range(len(counts))

    plt.bar(ticks,counts, align='center')
    plt.xticks(ticks, labels)

    plt.show()