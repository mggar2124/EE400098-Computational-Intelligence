import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def filter (signal, cutoff, sampling_frequency, order):
    sos = butter(order, cutoff, btype=type, analog=False, output='sos', fs=sampling_frequency)
    filtered = sosfiltfilt(sos, signal)
    return filtered_signal
    

