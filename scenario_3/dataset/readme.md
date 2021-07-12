### This folder "dataset" contains both datasets:
    - sleep_data_downsampling_AllSensorChannels_ lowfrequency_10HZ
    - sleep_lab_data
### We however downsampled the high resolution "sleep_lab_data" to 50Hz.

### To get the 50Hz sampled set, make sure the dataset folder has the 'sleep_lab_data' contained and set the 'generate' boolean in training_main.ipynb to True.
### The generate_dataset() method will be called and as the DatasetHandler is initialized as follows, the correct 50Hz folder is created and the data will be saved downsampled.
    - Dataset_Handler(dataset_folder='sleep_lab_data', target_hertz=50)
