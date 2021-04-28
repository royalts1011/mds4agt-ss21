import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_simple_data_loader():
    train_acc_meter = torch.Tensor(np.load('../../dataset/training/trainAccelerometer.npy'))
    train_grav = torch.Tensor(np.load('../../dataset/training/trainGravity.npy'))
    train_gyro = torch.Tensor(np.load('../../dataset/training/trainGyroscope.npy'))
    train_jin_acc = np.load('../../dataset/training/trainJinsAccelerometer.npy')
    train_jin_gyro = np.load('../../dataset/training/trainJinsGyroscope.npy')
    train_lin_acc = torch.Tensor(np.load('../../dataset/training/trainLinearAcceleration.npy'))
    train_mag_meter = np.load('../../dataset/training/trainMagnetometer.npy')
    train_ms_acc = np.load('../../dataset/training/trainMSAccelerometer.npy')
    train_ms_gyro = np.load('../../dataset/training/trainMSGyroscope.npy')

    train_labels = torch.Tensor(np.load('../../dataset/training/trainLabels.npy'))

    #print(train_acc_meter.shape)
    #print(train_grav.shape)
    #print(train_gyro.shape)
    #print(train_jin_acc.shape)
    #print(train_jin_gyro.shape)
    #print(train_lin_acc.shape)
    #print(train_mag_meter.shape)

    #print(train_ms_acc.shape)
    #print(train_ms_gyro.shape)

    # upsample some the small Tensors:
    train_jin_acc = torch.Tensor(np.repeat(train_jin_acc, 10, axis=1))
    train_jin_gyro = torch.Tensor(np.repeat(train_jin_gyro, 10, axis=1))
    train_mag_meter = torch.Tensor(np.repeat(train_mag_meter, 4, axis=1))

    #print(train_mag_meter.shape)

    #print(train_labels.shape)

    # concat all training Tensors:
    train_ls = [train_acc_meter, train_grav, train_gyro, train_jin_acc, train_jin_gyro, train_lin_acc, train_mag_meter]

    train_tens = torch.cat(train_ls, dim=2)

    #print(train_tens.shape)
    """
    change if activate when 2D or 1D
    """
    #train_tens = train_tens.view(2284,1, 21, 800)
    train_tens = train_tens.view(-1, 21, 800)

    # print(train_tens.shape)

    data_set_train = TensorDataset(train_tens, torch.LongTensor(train_labels.long()))

    #print(data_set_train[0])

    data_loader_train = DataLoader(data_set_train, batch_size=571, shuffle=True)

    return data_loader_train

def get_simple_eval_loader():
    eval_acc_meter = torch.Tensor(np.load('../../dataset/testing/testAccelerometer.npy'))
    eval_grav = torch.Tensor(np.load('../../dataset/testing/testGravity.npy'))
    eval_gyro = torch.Tensor(np.load('../../dataset/testing/testGyroscope.npy'))
    eval_jin_acc = np.load('../../dataset/testing/testJinsAccelerometer.npy')
    eval_jin_gyro = np.load('../../dataset/testing/testJinsGyroscope.npy')
    eval_lin_acc = torch.Tensor(np.load('../../dataset/testing/testLinearAcceleration.npy'))
    eval_mag_meter = np.load('../../dataset/testing/testMagnetometer.npy')
    # eval_ms_acc = np.load('../../dataset/testing/trainMSAccelerometer.npy')
    # eval_ms_gyro = np.load('../../dataset/testing/trainMSGyroscope.npy')

    eval_labels = torch.Tensor(np.load('../../dataset/testing/testLabels.npy'))

    # upsample some the small Tensors:
    eval_jin_acc = torch.Tensor(np.repeat(eval_jin_acc, 10, axis=1))
    eval_jin_gyro = torch.Tensor(np.repeat(eval_jin_gyro, 10, axis=1))
    eval_mag_meter = torch.Tensor(np.repeat(eval_mag_meter, 4, axis=1))

    # concat all training Tensors:
    eval_ls = [eval_acc_meter, eval_grav, eval_gyro, eval_jin_acc, eval_jin_gyro, eval_lin_acc, eval_mag_meter]

    eval_tens = torch.cat(eval_ls, dim=2)

    """
    change between 2D and 1D
    """
    # eval_tens = eval_tens.view(-1, 1, 21, 800)
    eval_tens = eval_tens.view(-1, 21, 800)

    # print(train_tens.shape)

    data_set_eval = TensorDataset(eval_tens, torch.LongTensor(eval_labels.long()))

    # print(data_set_train[0])

    data_loader_eval = DataLoader(data_set_eval, batch_size=1, shuffle=True)

    return data_loader_eval


if __name__ == "__main__":

    test_dataloader = get_simple_data_loader()