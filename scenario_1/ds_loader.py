import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
import os
from ds import Activity_Dataset
from torch.utils.data import Dataset, DataLoader
import torch
import random

class Dataset_Handler:

    train_dir_list = None
    test_dir_list = None
    excluded_file_endings = ("Labels.npy", "stacked_data.npy")

    def __init__(self, train_dir, test_dir, sampling_rate, seed):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.sampling = sampling_rate
        self.seed = seed
        self.generate_dir_lists()

    def generate_dir_lists(self):
        """
        Function to set lists to the content of the given training
        and testing directories
        """
        random.seed(self.seed)
        self.train_dir_list = sorted(os.listdir(self.train_dir))
        self.test_dir_list = sorted(os.listdir(self.test_dir))

        # TODO: RANDOM with seed not yet working

        # print(self.train_dir_list)
        # print(self.test_dir_list)
        # print()
        # # Apply random seed to sorted list of directory contents
        # random.shuffle(self.train_dir_list)
        # random.shuffle(self.test_dir_list)
        # print(self.train_dir_list)
        # print(self.test_dir_list)


    def get_folder_and_files(self, is_train):
        """
        Function to switch between training and testing paths
        and file directories
        Parameters
        ----------
        is_train : Boolean
            Indicator for training or testing
        """
        if is_train:
            return self.train_dir_list, self.train_dir
        else:
            return self.test_dir_list, self.test_dir


    def get_dataset(self, is_train=True, save=False):
        """
        Function to read in sensor files (.npy-files) and combining them into one stack.
        Can save data to the folder corresponding to is_train
        Parameters
        ----------
        is_train : Boolean
            Indicator for training or testing
        save : Boolean
            Indicator for saving the stacked data
        Returns
        ----------
        stacked_data : numpy array
            All combined sensory data
        """
        # Adapt variables to test/training scenario
        files, folder = self.get_folder_and_files(is_train)
        print('\n', "Computing the ", ('TESTING', 'TRAINING')[is_train] ," directory", '.'*5)
        data_list = []
        # For every .npy file
        for data in files:
            # Exclude non-sensor files
            if data.endswith(self.excluded_file_endings):
                print("Excluded:\t ", data)
                continue
            # Compute actual sensor file readings
            if data.endswith(".npy"):
                loaded_data = np.load(os.path.join(folder,data), allow_pickle=True)
                second_axis_size = np.size(loaded_data,axis=1)
                # Up-/Down-Sampling factor
                mult_factor = self.sampling / second_axis_size
                # Up-/Down-Sample
                resized_data = zoom(loaded_data, (1,mult_factor,1))
                data_list.append(resized_data)
                print("Computed:\t ", data)
                
        stacked_data = np.stack(data_list, axis = -1)
        
        if save:
            np.save(os.path.join(folder,"stacked_data.npy"),stacked_data)

        return stacked_data

    def get_dataloader(self, is_train=True, batch_size=32, num_workers=0):
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
        """
        # Adapt variables to test/training scenario
        files, folder = self.get_folder_and_files(is_train)

        for data in files:
            if data.endswith("Labels.npy"):
                labels = np.load(os.path.join(folder,data), allow_pickle=True)
            if data.endswith("stacked_data.npy"):
                stacked_data = np.load(os.path.join(folder,data), allow_pickle=True)

        dataset = Activity_Dataset(stacked_data,labels)
        
        ds_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        return ds_loader