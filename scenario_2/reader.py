from os.path import join
import pandas as pd

def csv_to_dataframe(folder):
    """
        Fast pandas conversion to DataFrame.
        Return DataFrame format 
    """
    # No skipped rows as the first row is the heder of the DataFrame
    accelerometer = pd.read_csv(join(folder,'Accelerometer.csv'))
    gyroscope = pd.read_csv(join(folder,'Gyroscope.csv'))
    return accelerometer, gyroscope


def csv_to_array(folder):
    """
        This function can be used from externa to translate a CSV
        into a numpy array.
    """
    acc, gyro = csv_to_dataframe(folder)
    return acc.to_numpy(), gyro.to_numpy()

track_dir = 'dataset/track1'

# Pandas version *FAST*
acc, gyro = csv_to_dataframe(folder=track_dir)
print(acc.head(5))
print(gyro.head(5))

acc = acc.to_numpy()
gyro = gyro.to_numpy()
print(acc[0:5,:])
print(gyro[0:5,:])
