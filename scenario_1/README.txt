this document provides a brief explanation of how to use the code.

 1) First provide the path to the dataset in the variable path_to_data in the 8th row in utils.py
    (in this directory there should be te two folders training and testing from which the methods in utils.py build the
    pytorch DataLoader)

 2) Execute the training.py script
    (if the code is executed on a machine with cuda installed, the graphics chip is used to speed up the training.
    Training on the CPU can take some time)


 Further options in the training.py file:

 1) We also provide a trained model. This can be loaded if the "load_model" flag is set to true. Then you have to
    provide the path to the model in the "path_to_model" variable in the training.py file.
    (Note that the given path is to a model located in the "model" directory that is created when you save a model for
    the first time)

 2) You can sava a trained model by setting the "save_model" flag to true. If the directory "models" does not exist at
    the given position a new one is created and the models are saved there with the achieved accuracy as part of their
    name.


 The model definition is located in the net.py file.