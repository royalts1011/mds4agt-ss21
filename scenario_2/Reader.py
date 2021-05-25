import pandas as pd
import os

def csv_reader(folder='dataset/track1'):
    accelerometer = pd.read_csv(os.path.join(folder,'Accelerometer.csv'))
    gyroscope = pd.read_csv(os.path.join(folder,'Gyroscope.csv'))
    return accelerometer, gyroscope


folder = 'dataset/track1'
acc, gyro = csv_reader(folder=folder)