import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
import os
from ds import Activity_Dataset
from torch.utils.data import Dataset, DataLoader


class Config():
    DATABASE_FOLDER_TRAIN = str(Path("../../dataset/training/"))
    DATABASE_FOLDER_TEST = str(Path("../../dataset/testing/"))
    SAMPLING_RATE = 800

def get_dataset(save=False, folder=Config.DATABASE_FOLDER_TRAIN):
    data_list = []
    for data in sorted(os.listdir(folder)):
        if data.endswith("Labels.npy"):
            continue
        if data.endswith("stacked_data.npy"):
            continue
        if data.endswith(".npy"):
            loaded_data = np.load(os.path.join(folder,data), allow_pickle=True)
            second_axis_size = np.size(loaded_data,axis=1)
            mult_factor = Config.SAMPLING_RATE / second_axis_size

            resized_data = zoom(loaded_data, (1,mult_factor,1))
            data_list.append(resized_data)
            
    stacked_data = np.stack(data_list, axis = -1)
    
    if save:
        np.save(os.path.join(folder,"stacked_data.npy"),stacked_data)

    return stacked_data

def get_dataloader(is_train=True, batch_size=32, num_workers=0):
    if is_train:
        folder = Config.DATABASE_FOLDER_TRAIN
    else:
        folder = Config.DATABASE_FOLDER_TEST

    for data in sorted(os.listdir(folder)):
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

get_dataset(save=True, folder=Config.DATABASE_FOLDER_TEST)
