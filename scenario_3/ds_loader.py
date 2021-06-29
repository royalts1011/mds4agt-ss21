import numpy as np
from scipy.ndimage import zoom
import os
from torch.utils.data import Dataset, DataLoader
from os.path import join
from os import listdir
from pathlib import Path
import pandas as pd
from datetime import datetime
from ds import Sleep_Stage_Dataset


class Dataset_Handler:
    modality1 = ["O1M2", "O2M1", "C4M1", "C3M2", "F4M1", "F3M2"]
    modality2 = ["LEOGM2", "REOGM1"]
    modality3 = ["EMG"]
    modality4 = ["BeinLi", "BeinRe"]

    label_dict = {
        'WK': 0,
        'N1': 1,
        'N2': 2,
        'N3': 3,
        'REM': 4
    }

    def __init__(self, dataset_folder, target_hertz):
        self.dataset_dir = join(Path('./dataset'), dataset_folder)
        self.patient_names = listdir(self.dataset_dir)
        # Uses the first entry of patient (name) folder to get the .csv filenames
        # These .csv file names will be the same for every other patient 
        self.sensor_files = listdir(join(self.dataset_dir, self.patient_names[0]))
        self.hertz = target_hertz
        self.samples_per_30sec = target_hertz * 30

    def channel_csv_to_array(self, file_path):
        """
            This function extracts the first row of csv to calculate the time difference and returns
            the data as a numpy array as well as the elapsed seconds during the sleep

        :param: file_path:
            The path to a sensor csv file
        """
        df = pd.read_csv(file_path, skiprows=1, names=['value'])
        arr = df.to_numpy().flatten()

        # Get first row, as it contains the start and end of sleep
        first_row = pd.read_csv(file_path, nrows=1, names=['start', 'end']).values.tolist()[0]

        # declare time format
        FMT = '%H:%M:%S'
        # Calculate time elapsed
        tdelta = datetime.strptime(first_row[1], FMT) - datetime.strptime(first_row[0], FMT)

        # return sensor raw sensor data and time elapsed
        return arr, tdelta.seconds

    def get_labels(self, patient, target_amount):
        df_labels = pd.read_csv(join(self.dataset_dir, patient, "SleepStaging.csv"))
        # Get column 'Schlafstadium' as a list 
        sleep_stage_list = df_labels["Schlafstadium"].tolist()
        # Convert description to index
        sleep_stage_list = [self.label_dict[stage] for stage in sleep_stage_list]
        # Trim list to same length as modality data
        sleep_stage_list = sleep_stage_list[:target_amount]
        return np.array(sleep_stage_list)

    def get_modality_data(self, modality):
        # Dictionary containing the patients name and data corresponding to the given modality
        dict_modality = {}
        for patient in self.patient_names:
            all_channels = []
            for channel in modality:
                channel_csv = list(filter(lambda x: channel in x, self.sensor_files))[0]
                # if channel_csv is not self.sleepStaging_csv:
                channel_path = join(self.dataset_dir, patient, channel_csv)
                channel_data, seconds = self.channel_csv_to_array(channel_path)

                channel_freq = int(len(channel_data) / seconds)

                # Up-/Down-Sampling factor
                mult_factor = self.hertz / channel_freq

                # Up-/Down-Sample
                resized_data = zoom(channel_data, zoom=mult_factor, order=0)

                # Cut last values if total samples if not multiple from samples_per_frame
                smooth_factor = self.samples_per_30sec * int(len(resized_data) / self.samples_per_30sec)
                resized_data = resized_data[:smooth_factor]

                amount_time_windows = int(len(resized_data) / self.samples_per_30sec)
                # Split into time windows
                resized_data = np.array_split(resized_data, amount_time_windows)
                # Append data to all channels of modality
                all_channels.append(resized_data)

            # convert to a numpy array
            all_channels = np.array(all_channels)

            # TODO: Where to save/integrate labels
            labels = self.get_labels(patient, all_channels.shape[1])

            # TODO: How to concatenate the patients and labels instead of dictionary
            dict_modality[patient] = (all_channels, labels)

            # concat_modality = np.concatenate(all_channels, axis=1, out=None)

        return dict_modality

    def generate_dataset(self):
        """
            Invoke CSV-readings and concatenate for baseline model.
        
        """

        dict_modality1 = self.get_modality_data(self.modality1)
        dict_modality2 = self.get_modality_data(self.modality2)
        dict_modality3 = self.get_modality_data(self.modality3)
        dict_modality4 = self.get_modality_data(self.modality4)

        labels_1 = [y for (_, y) in dict_modality1.values()]
        concat_labels = np.concatenate((labels_1), axis=0, out=None)

        all_1 = [x for (x, _) in dict_modality1.values()]
        all_2 = [x for (x, _) in dict_modality2.values()]
        all_3 = [x for (x, _) in dict_modality3.values()]
        all_4 = [x for (x, _) in dict_modality4.values()]

        # Concatenate in frame axis
        concat_channel_modality1 = np.concatenate((all_1), axis=1, out=None)
        concat_channel_modality2 = np.concatenate((all_2), axis=1, out=None)
        concat_channel_modality3 = np.concatenate((all_3), axis=1, out=None)
        concat_channel_modality4 = np.concatenate((all_4), axis=1, out=None)

        concat_baseline = np.concatenate(
            (concat_channel_modality1,
             concat_channel_modality2,
             concat_channel_modality3,
             concat_channel_modality4), axis=0, out=None)

        # Saving directory
        save_dir = join(Path('./dataset'), str(self.hertz))
        os.makedirs(save_dir, exist_ok=True)

        np.save(join(save_dir, "labels.npy"), concat_labels)

        np.save(join(save_dir, "modality1.npy"), concat_channel_modality1)
        np.save(join(save_dir, "modality2.npy"), concat_channel_modality2)
        np.save(join(save_dir, "modality3.npy"), concat_channel_modality3)
        np.save(join(save_dir, "modality4.npy"), concat_channel_modality4)

        np.save(join(save_dir, "baseline.npy"), concat_baseline)

        return True

    def get_dataloader(self, herz=50, modality="baseline.npy", batch_size=32, num_workers=0):
        """
        Function to create dataset and dataloaders from labels and stacked data (all sensors combined)
        Parameters
        ----------
        is_train : Boolean
            Indicator for training or testing
        batch_size : int
            Sets dataloader batch size
        num_workers : int
            Amount of additional workers being used
        Returns
        ----------
        ds_loader : DataLoader
            All combined sensory data in the dataloader
            :param load_dir:
        """
        load_dir = join(Path('./dataset'), str(herz))

        labels = np.load(os.path.join(load_dir, "labels.npy"), allow_pickle=True)
        data = np.load(os.path.join(load_dir, modality), allow_pickle=True)

        data = data.transpose(1, 2, 0)

        N = len(data)
        n_70 = int(round(.7*N))

        data_train = data[:n_70]
        data_test = data[n_70:]
        label_train = labels[:n_70]
        label_test = labels[n_70:]
        dataset_train = Sleep_Stage_Dataset(data_train, label_train)
        dataset_test = Sleep_Stage_Dataset(data_test, label_test)

        train_dl = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        test_dl = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        return train_dl, test_dl


# if __name__ == "__main__":
    # from ds_loader import Dataset_Handler
    # dsh = Dataset_Handler(dataset_folder='sleep_lab_data', target_hertz=50)
    # dsh.get_dataloader()

    # initialise Dataset_Handler to build Dataset and Dataloader
    # dsh = Dataset_Handler(dataset_folder='sleep_data_downsampling_AllSensorChannels_ lowfrequency_10HZ', target_hertz=10)
    # dsh = Dataset_Handler(dataset_folder='sleep_lab_data', target_hertz=50)

    # dsh.get_dataset()
    # print(dsh.sensor_files)
