
import numpy as np
from numpy.linalg import norm
from scipy import signal



def acc_lowpass_filter(acc_data, sample_freq):
    """
    lowpass filter the acc data

    track_1 ---> 2/1
    track_2 ---> 3/1
    track_3 ---> 2/6
    track geb. 64 ---> 2/2

    :param acc_data:  raw acceleration data
    :param sample_freq: sampling rate of the data
    :return: lowpass filtered acceleration data
    """
    # sos: second-order sections (‘sos’) should be used for general-purpose filtering.
    # fs: sampling rate
    # btype: which kind of filter
    # N: 2 Lowpass Filter-order --> high order step function, low order more smooth function
    # Wn: ?????
    sos = signal.butter(N=2, Wn=2, btype='lowpass', analog=False, output='sos', fs=sample_freq)
    prepro_acc = signal.sosfiltfilt(sos, acc_data, axis=0)

    return prepro_acc


def peakfinder(acc_data, sample_freq):
    """
    normalize data and find the peaks in the accelerometer data which is interpreted as steps
    :param acc_data: acceleration data (should be preprocessed at this point)
    :param sample_freq: sampling rate of the data
    :return: number of peaks, peakindex in the accelerometer data(step index), normalzied&combined acceleration vector
    """
    combined_acc = norm(acc_data, axis=1)

    # normalize combined_acc data
    normalised_acc = normalize_data(combined_acc)


    # width: number of samples per peak
    # wlen: window length in samples that optionally limits the evaluated area for each peak
    # peak_height: min height of a peak
    # prominence: measures how much a peak stands out from the surrounding baseline of the signal
    peak_height = normalised_acc.mean()
    peaks = signal.find_peaks(normalised_acc, height=peak_height, prominence=0.1, width=1, wlen=sample_freq)

    return len(peaks[0]), peaks[0], normalised_acc

def normalize_data(data):
    '''
    this function normalizes the given input to [0...1]
    :param data:
    :return: input data got normalized in area [0...1]
    '''
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def remove_time_dim(data, sample_freq):
    '''

    :param data:
    :param sample_freq:
    :return:
    '''
    return data[:,1:]