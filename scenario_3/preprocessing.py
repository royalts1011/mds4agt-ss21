import numpy as np

def upsample_N1(data, labels, label_dict, factor=4):
    """This function upsamples the N1 stage in the data.
        (Independent sleep_stage upsampling could be implemented by gicing a sleep_stage parameter)

    Args:
        data : Multidimensional modality data
        labels : Corresponding labels
        label_dict (dictionary) : Dictionary to decode string sleep stage to integer label 
        factor (int, optional) : Multiplication to reach same amount of frequency. Defaults to 4.

    Returns:
        new data with additional N1 stages and new labels. Noise is applied to additional N1 stages.
    """        
    target_label = label_dict['N1']
    new_data = []
    # Go through every channel of the modality
    for j in range(data.shape[0]):
        channel_n1s = []
        # Go through all data frames = amount of sleep stage labels
        for i in range(data.shape[1]):
            if labels[i] == target_label:
                # Append N1 frame (factor-1) times
                # channel_n1s.extend([data[j][i] for x in range(factor-1)])
                channel_n1s.extend([ data[j][i] ] *(factor-1) )
        
        # add noise to all additional n1 windows
        noisy_n1s = add_noise(channel_n1s, noise_std=.2)

        # get current loop channel
        channel = data[j]

        # append/concatenate all noisy N1 frames to channel
        new_channel = np.concatenate((channel, noisy_n1s))

        # Create and append to the new modality data
        new_data.append( new_channel )

    new_data = np.array(new_data)

    # Difference shows added N1 frames
    added_n1s = new_data.shape[1] - data.shape[1]
    # Add as many N1 labels to the end as frames were added
    new_labels = np.append(labels, [label_dict['N1'] for x in range(added_n1s)])

    return new_data, new_labels

def add_noise(frame_list, noise_std):
    """Adds noise to each frame in a list of frames

    Args:
        frame_list : List of 'pure' frames (list of arrays)
        noise_std ([float]): The standard deviation for noise generation.
                                0.2 or 0.4 recommended

    Returns:
        Numpy array of frames with noise applied to them
    """        
    noisy_list = []
    for frame in frame_list:
        # Create new noise in each iteration
        noise = np.random.normal(0, noise_std, frame.shape)

        new_frame = frame + noise
        noisy_list.append( new_frame )

    return np.array(noisy_list)