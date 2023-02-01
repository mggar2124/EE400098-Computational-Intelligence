import scipy.io as spio
from scipy.stats import median_abs_deviation
from scipy.signal import butter, find_peaks, lfilter
from scipy.optimize import dual_annealing

import numpy as np
import matplotlib.pyplot as plt
from operator import sub
from collections import defaultdict
# from numpy.random import seed

from statistics import median
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA


# seed(25)

# bandpass
def btrwth_bpass(sample_freq, inputs, lower_bounds, higher_bounds, order=2):
    lower = lower_bounds / (0.5 * sample_freq)
    higher = higher_bounds / (0.5 * sample_freq)
    x, y = butter(order, [lower, higher],  btype='band')
    return lfilter(x, y, inputs)

# initial preprocesses/reference
def preproc_D1(inputs, window_of_spike, lower_cutoff, upper_cutoff, peaks, wlen, sample_freq, idx_train, class_train):
    D1_spikes = []
    identified_classes = []
    idx_srt = []
    indetified_peaks = []
    duplicate_counter = 0

    bandpass_D1 = btrwth_bpass(
        sample_freq, inputs, lower_cutoff, upper_cutoff)
    peak_idx = find_peaks(
            bandpass_D1, prominence=(peaks*median_abs_deviation(bandpass_D1)), wlen=wlen)[0]

    for peak in peak_idx:

        if len(idx_train[idx_train < peak]) != 0:
            peak_mt = max(idx_train[idx_train < peak])
        else:
            continue

        if peak_mt in idx_srt:
            duplicate_counter = duplicate_counter + 1
            continue

        identified_spikes_list = bandpass_D1[peak - window_of_spike[0]: peak + window_of_spike[1]]
        indetified_peaks.append(peak)

        identified_classes.append(
            class_train[np.where(idx_train == peak_mt)[0][0]])

        idx_srt.append(peak_mt)
        D1_spikes.append(identified_spikes_list)

    rem_peaks = len(idx_train) - len(D1_spikes)

    return D1_spikes, identified_classes, idx_srt, indetified_peaks, rem_peaks, duplicate_counter

# preprocessing steps post initialisation
def preprocessing(inputs, window_of_peaks, lower_cutoff, upper_cuttoff, peaks, wlen, sampFreq):

    preproc_spikes = []
    preproc_idx = []
    preproc_peaks = []

    preproc_bandpass = btrwth_bpass(
        sampFreq, inputs, lower_cutoff, upper_cuttoff)

    peak_idx = find_peaks(preproc_bandpass, prominence=(
        peaks * median_abs_deviation(preproc_bandpass)), wlen=wlen)[0]

    for peak in peak_idx:
        index = peak - window_of_peaks[0]
        spike = preproc_bandpass[peak - window_of_peaks[0]:peak + window_of_peaks[1]]

        if len(spike) == (window_of_peaks[0] + window_of_peaks[1]):
            preproc_spikes.append(spike)
            preproc_idx.append(index)
            preproc_peaks.append(peak)

    return preproc_spikes, preproc_idx, preproc_peaks

# spike indexing
def rec_sp_idx(peaks, classes, rT):

    classal_rT = {c: t for c, t in zip(range(1, 6), rT)}

    idx_array = [round(peaks[i] - classal_rT[classes[i]])
                 for i in range(len(peaks))]

    return idx_array

# rT cacl
def rftc(identified_peaks, idx_srt, classes):

    rT = list(map(sub, identified_peaks, idx_srt))

    classal_avg = {c: sum(rT[i] for i in indixes) / len(indixes) for c, indixes in (
        (x, [i for i, z in enumerate(classes) if z == x]) for x in set(classes))}

    rT_avg = [classal_avg.get(i, 0) for i in range(1, 6)]

    return rT_avg

# filtering optimisation
def filt_opt(bounds):

    lower_cutoff, upper_cutoff, x_prom, wlen, sampling_freq = bounds
    if for_D1 == 0:
        spikes_train, i, p = preprocessing(
            inputs, window_of_spikes, lower_cutoff, upper_cutoff, x_prom, wlen, sampling_freq)
        score = abs(target-len(spikes_train))

    if for_D1 == 1:
        t, l, s, c, rem_peaks, duplicates = preproc_D1(
            inputs_D1, window_of_spikes, lower_cutoff, upper_cutoff, x_prom, wlen, sampling_freq, idx_D1, classal_D1)
        score = rem_peaks + duplicates
    return score

# optimising classifiers
def opt_classifiers(bounds):
    rise, fall, k, p, PCAlevel = bounds
    k = round(k)
    p = round(p)

    spikes_D1, classesD1, b, l, m, f = preproc_D1(
            inputs_D1, (round(rise), round(fall)), lower_cutoff, upper_cutoff, prominance, wlen, sampling_freq, idx_D1, classal_D1)

    comp_spikes = PCA(n_components=PCAlevel).fit_transform(spikes_D1)
    spikes_train, spikes_test, class_train, testingClass = train_test_split(
        comp_spikes, classesD1, test_size=0.2)

    knn = KNeighborsClassifier(n_neighbors=k, p=p)
    knn.fit(spikes_train, class_train)
    class_pred = knn.predict(spikes_test)

    acc = metrics.accuracy_score(testingClass, class_pred)
    cost = target - acc

    return cost

# final count collation
def classal_counters(input_array):
    counts = defaultdict(int)
    for elements in input_array:
        counts[elements] += 1
    return counts[1], counts[2], counts[3], counts[4], counts[5]


# loading
train_mat = spio.loadmat('D1.mat', squeeze_me=True)
inputs_D1 = train_mat['d']
idx_D1 = train_mat['Index']
classal_D1 = train_mat['Class']

D2_inputs = spio.loadmat('D2.mat', squeeze_me=True)
D2_array = D2_inputs['d']
D3_inputs = spio.loadmat('D3.mat', squeeze_me=True)
D3_array = D3_inputs['d']
D4_inputs = spio.loadmat('D4.mat', squeeze_me=True)
D4_array = D4_inputs['d']


# D1 operations
anneal_iters = 11
target = 1.0
window_of_spikes = [15, 25]
bounds = [[25, 150], [1750, 3000], [3, 6], [25, 150], [25000, 50000]]
for_D1 = 1  # start with D1 to use for the next tasks


D1_anneal_param = dual_annealing(
    filt_opt, bounds, maxfun=anneal_iters)


bounds = [[5, 50], [5, 50], [1, 20], [1, 20], [0.5, 0.99]]

sampling_freq = D1_anneal_param.x[4]
wlen = D1_anneal_param.x[3]
prominance = D1_anneal_param.x[2]
upper_cutoff = D1_anneal_param.x[1]
lower_cutoff = D1_anneal_param.x[0]

D1_knn_param = dual_annealing(
    opt_classifiers, bounds, maxfun=anneal_iters)

comp_fct = D1_knn_param.x[4]
p = round(D1_knn_param.x[3])
k = round(D1_knn_param.x[2])
fT = round(D1_knn_param.x[1])
rT = round(D1_knn_param.x[0])


KNN = KNeighborsClassifier(n_neighbors=k, p=p)
window_of_spikes = [rT, fT]

D1_spikes, D1_classes, idx_srt, identified_peaks, rem_peaks, duplicates = preproc_D1(
    inputs_D1, window_of_spikes, lower_cutoff, upper_cutoff, prominance, wlen, sampling_freq, idx_D1, classal_D1)
classal_rT = rftc(identified_peaks, idx_srt, D1_classes)
spike_train, spike_test, class_train, class_test = train_test_split(
    D1_spikes, D1_classes, test_size=0.05)
KNN.fit(spike_train, class_train)

predicted_class_test = KNN.predict(spike_test)

performance_metrics = metrics.classification_report(
    class_test, predicted_class_test, digits=4)

# D2 operations
inputs = D2_array
for_D1 = 0
bounds = [[25, 150], [1500, 3000], [3, 6], [25, 150], [25000, 50000]]
target = 2365


D1_anneal_param = dual_annealing(
    filt_opt, bounds, maxfun=anneal_iters)
sampling_freq = D1_anneal_param.x[4]
wlen = D1_anneal_param.x[3]
prominance = D1_anneal_param.x[2]
upper_cutoff = D1_anneal_param.x[1]
lower_cutoff = D1_anneal_param.x[0]




ext_D2_spikes, ext_D2_idx, ext_D2_peaks = preprocessing(
    inputs, window_of_spikes, lower_cutoff, upper_cutoff, prominance, wlen, sampling_freq)

D2_classes = KNN.predict(ext_D2_spikes)
D2_idx = rec_sp_idx(ext_D2_peaks, D2_classes, classal_rT)

# D3 operations
inputs = D3_array
for_D1 = 0
bounds = [[25, 150], [1500, 3000], [3, 6], [25, 150], [25000, 50000]]
target = 3425


D3_FilteringParameters = dual_annealing(
    filt_opt, bounds, maxfun=anneal_iters)

sampling_freq = D3_FilteringParameters.x[4]
wlen = D3_FilteringParameters.x[3]
prominance = D3_FilteringParameters.x[2]
upper_cutoff = D3_FilteringParameters.x[1]
lower_cutoff = D3_FilteringParameters.x[0]

ext_D3_spikes, ext_D3_idx, ext_D3_peaks = preprocessing(
    inputs, window_of_spikes, lower_cutoff, upper_cutoff, prominance, wlen, sampling_freq)

D3_class = KNN.predict(ext_D3_spikes)
D3_idx = rec_sp_idx(ext_D3_peaks, D3_class, classal_rT)

# D4 operations
ext_D4_spikes, ext_D4_idx, ext_D4_peaks = preprocessing(
    D4_array, window_of_spikes, lower_cutoff, upper_cutoff, prominance, wlen, sampling_freq)

D4_classes = KNN.predict(ext_D4_spikes)

D4_idx = rec_sp_idx(ext_D4_peaks, D4_classes, classal_rT)


# outputing
class_one, class_two, class_three, class_four, class_five = classal_counters(D2_classes)
print("num_D2: ")
print("Class One:", class_one, " Class Two: ", class_two, " Class Three: ",
      class_three, " Class Four: ", class_four, " Class Five: ", class_five)

class_one, class_two, class_three, class_four, class_five = classal_counters(D3_class)
print("num_D3: ")
print("Class One:", class_one, " Class Two: ", class_two, " Class Three: ",
      class_three, " Class Four: ", class_four, " Class Five: ", class_five)

class_one, class_two, class_three, class_four, class_five = classal_counters(D4_classes)
print("num_D4: ")
print("Class One:", class_one, " Class Two: ", class_two, " Class Three: ",
      class_three, " Class Four: ", class_four, " Class Five: ", class_five)
print('Finished.')


#output file creation
output_path = 'results_task_4.mat'
spio.savemat(output_path, mdict={'d': D4_array.tolist(
), 'Index': D4_idx, 'Class': D4_classes.tolist()})

output_path = 'results_task_3.mat'
spio.savemat(output_path, mdict={'d': D3_array.tolist(
), 'Index': D3_idx, 'Class': D3_class.tolist()})

output_path = 'results_task_2.mat'
spio.savemat(output_path, mdict={'d': D2_array.tolist(
), 'Index': D2_idx, 'Class': D2_classes.tolist()})




