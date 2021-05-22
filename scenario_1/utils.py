import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.ndimage import zoom
import os

# path to the folder which hold the folders "training" and "testing"
path_to_data = '../../dataset/'


def get_simple_data_loader(cuda=False):
    """
    arrays are loaded from the the specified folder "path_to_data"
    data with smaller sampling rates are upsampeld to 800Hz with the same values
    :param cuda:
    :return data loader for training data:
    """
    train_acc_meter = torch.Tensor(np.load(path_to_data + 'training' + os.sep + 'trainAccelerometer.npy'))
    train_grav = torch.Tensor(np.load(path_to_data + 'training' + os.sep + 'trainGravity.npy'))
    train_gyro = torch.Tensor(np.load(path_to_data + 'training' + os.sep + 'trainGyroscope.npy'))
    train_jin_acc = np.load(path_to_data + 'training' + os.sep + 'trainJinsAccelerometer.npy')
    train_jin_gyro = np.load(path_to_data + 'training' + os.sep + 'trainJinsGyroscope.npy')
    train_lin_acc = torch.Tensor(np.load(path_to_data + 'training' + os.sep + 'trainLinearAcceleration.npy'))
    train_mag_meter = np.load(path_to_data + 'training' + os.sep + 'trainMagnetometer.npy')
    train_ms_acc = np.load(path_to_data + 'training' + os.sep + 'trainMSAccelerometer.npy')
    train_ms_gyro = np.load(path_to_data + 'training' + os.sep + 'trainMSGyroscope.npy')

    train_labels = torch.Tensor(np.load(path_to_data + 'training' + os.sep + 'trainLabels.npy'))

    # upsample the small Tensors for more speed with numpy:
    train_jin_acc = torch.Tensor(np.repeat(train_jin_acc, 10, axis=1))
    train_jin_gyro = torch.Tensor(np.repeat(train_jin_gyro, 10, axis=1))
    train_mag_meter = torch.Tensor(np.repeat(train_mag_meter, 4, axis=1))

    # upsampling not possible with np.repeat because length ist not dividable without rest
    train_ms_acc = torch.Tensor(zoom(train_ms_acc, (1, 200/67, 1)))
    train_ms_gyro = torch.Tensor(zoom(train_ms_gyro, (1, 200/67, 1)))

    # concat all training Tensors:
    train_ls = [train_acc_meter, train_grav, train_gyro, train_jin_acc, train_jin_gyro, train_lin_acc, train_mag_meter,
                train_ms_acc, train_ms_gyro]

    train_tens = torch.cat(train_ls, dim=2)

    # change if activate when 2D or 1D
    train_tens = train_tens.view(2284, 1, 27, 800)
    # train_tens = train_tens.view(-1, 27, 800)

    # print(train_tens.shape)
    if cuda:
        data_set_train = TensorDataset(train_tens.cuda(), torch.LongTensor(train_labels.long()).cuda())
    else:
        data_set_train = TensorDataset(train_tens, torch.LongTensor(train_labels.long()))

    data_loader_train = DataLoader(data_set_train, batch_size=32, shuffle=True)

    return data_loader_train


def get_simple_eval_loader(cuda=False):
    """
    arrays are loaded from the the specified folder "path_to_data"
    data with smaller sampling rates are upsampled to 800Hz with the same values
    :param cuda:
    :return data loader for testdata:
    """
    eval_acc_meter = torch.Tensor(np.load(path_to_data + 'testing' + os.sep + 'testAccelerometer.npy'))
    eval_grav = torch.Tensor(np.load(path_to_data + 'testing' + os.sep + 'testGravity.npy'))
    eval_gyro = torch.Tensor(np.load(path_to_data + 'testing' + os.sep + 'testGyroscope.npy'))
    eval_jin_acc = np.load(path_to_data + 'testing' + os.sep + 'testJinsAccelerometer.npy')
    eval_jin_gyro = np.load(path_to_data + 'testing' + os.sep + 'testJinsGyroscope.npy')
    eval_lin_acc = torch.Tensor(np.load(path_to_data + 'testing' + os.sep + 'testLinearAcceleration.npy'))
    eval_mag_meter = np.load(path_to_data + 'testing' + os.sep + 'testMagnetometer.npy')
    eval_ms_acc = np.load(path_to_data + 'testing' + os.sep + 'testMSAccelerometer.npy')
    eval_ms_gyro = np.load(path_to_data + 'testing' + os.sep + 'testMSGyroscope.npy')

    eval_labels = torch.Tensor(np.load(path_to_data + 'testing' + os.sep + 'testLabels.npy'))

    # upsample some the small Tensors:
    eval_jin_acc = torch.Tensor(np.repeat(eval_jin_acc, 10, axis=1))
    eval_jin_gyro = torch.Tensor(np.repeat(eval_jin_gyro, 10, axis=1))
    eval_mag_meter = torch.Tensor(np.repeat(eval_mag_meter, 4, axis=1))

    eval_ms_acc = torch.Tensor(zoom(eval_ms_acc, (1, 200 / 67, 1)))
    eval_ms_gyro = torch.Tensor(zoom(eval_ms_gyro, (1, 200 / 67, 1)))

    # concat all training Tensors:
    eval_ls = [eval_acc_meter, eval_grav, eval_gyro, eval_jin_acc, eval_jin_gyro, eval_lin_acc, eval_mag_meter,
               eval_ms_acc, eval_ms_gyro]

    eval_tens = torch.cat(eval_ls, dim=2)

    # change between 2D and 1D
    eval_tens = eval_tens.view(-1, 1, 27, 800)
    # eval_tens = eval_tens.view(-1, 27, 800)

    if cuda:
        data_set_eval = TensorDataset(eval_tens.cuda(), torch.LongTensor(eval_labels.long()).cuda())
    else:
        data_set_eval = TensorDataset(eval_tens, torch.LongTensor(eval_labels.long()))

    data_loader_eval = DataLoader(data_set_eval, batch_size=1, shuffle=False)

    return data_loader_eval


if __name__ == "__main__":

    test_dataloader = get_simple_data_loader()

