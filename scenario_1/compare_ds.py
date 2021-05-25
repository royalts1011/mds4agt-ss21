import numpy as np
from pathlib import Path
import os

class Config():
    DATABASE_FOLDER_TRAIN = str(Path("../../dataset/training/"))
    DATABASE_FOLDER_TEST = str(Path("../../dataset/testing/"))


for data in os.listdir(Config.DATABASE_FOLDER_TRAIN):
    if data.endswith("Labels.npy"):
        labels = np.load(os.path.join(Config.DATABASE_FOLDER_TRAIN,data), allow_pickle=True)
    if data.endswith("jesse_stacked_data.npy"):
        jesse_stacked_train = np.load(os.path.join(Config.DATABASE_FOLDER_TRAIN,data), allow_pickle=True)
        continue
    if data.endswith("stacked_data.npy"):
        not_working_train_stacked = np.load(os.path.join(Config.DATABASE_FOLDER_TRAIN,data), allow_pickle=True)
        continue



for data in os.listdir(Config.DATABASE_FOLDER_TEST):
    if data.endswith("Labels.npy"):
        labels = np.load(os.path.join(Config.DATABASE_FOLDER_TEST,data), allow_pickle=True)
    if data.endswith("jesse_stacked_data.npy"):
        jesse_stacked_test = np.load(os.path.join(Config.DATABASE_FOLDER_TEST,data), allow_pickle=True)
        continue
    if data.endswith("stacked_data.npy"):
        not_working_test_stacked = np.load(os.path.join(Config.DATABASE_FOLDER_TEST,data), allow_pickle=True)
        continue