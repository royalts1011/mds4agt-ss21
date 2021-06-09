from os.path import join
import pandas as pd
import math


def csv_to_dataframe(folder, delimiter=','):
    """
        Fast pandas conversion to DataFrame.
        Return DataFrame format 
    """
    # No skipped rows as the first row is the header of the DataFrame
    accelerometer = pd.read_csv(join(folder, 'Accelerometer.csv'), delimiter=delimiter)
    gyroscope = pd.read_csv(join(folder, 'Gyroscope.csv'), delimiter=delimiter)
    return accelerometer, gyroscope


def csv_to_array(folder, delimiter):
    """
    :param: data:
    This function can be used from externa to translate a CSV
    into a numpy array.
    """
    acc, gyro = csv_to_dataframe(folder, delimiter)
    return acc.to_numpy(), gyro.to_numpy()


def get_sample_freq(data):
    """
         :return: rounded sample frequency in Hz
    """
    time = data[len(data) - 2][0]
    freq = len(data) / time  # if needed this is the original freq

    return int(math.ceil(freq / 100.0)) * 100
