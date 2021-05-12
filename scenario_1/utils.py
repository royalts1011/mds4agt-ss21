import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.ndimage import zoom


def get_simple_data_loader(cuda=False):
    train_acc_meter = torch.Tensor(np.load('../../dataset/training/trainAccelerometer.npy'))
    train_acc_meter = train_acc_meter.view(-1, 3, 800)
    
    train_grav = torch.Tensor(np.load('../../dataset/training/trainGravity.npy'))
    train_grav = train_grav.view(-1, 3, 800)

    train_gyro = torch.Tensor(np.load('../../dataset/training/trainGyroscope.npy'))
    train_gyro = train_gyro.view(-1, 3, 800)

    train_jin_acc = torch.Tensor(np.load('../../dataset/training/trainJinsAccelerometer.npy'))
    train_jin_acc = train_jin_acc.view(-1, 3, 80)

    train_jin_gyro = torch.Tensor(np.load('../../dataset/training/trainJinsGyroscope.npy'))
    train_jin_gyro = train_jin_gyro.view(-1, 3, 80)

    train_lin_acc = torch.Tensor(np.load('../../dataset/training/trainLinearAcceleration.npy'))
    train_lin_acc = train_lin_acc.view(-1, 3, 800)

    train_mag_meter = torch.Tensor(np.load('../../dataset/training/trainMagnetometer.npy'))
    train_mag_meter = train_mag_meter.view(-1, 3, 200)

    train_ms_acc = torch.Tensor(np.load('../../dataset/training/trainMSAccelerometer.npy'))
    train_ms_acc = train_ms_acc.view(-1, 3, 268)

    train_ms_gyro = torch.Tensor(np.load('../../dataset/training/trainMSGyroscope.npy'))
    train_ms_gyro = train_ms_gyro.view(-1, 3, 268)

    train_labels = torch.Tensor(np.load('../../dataset/training/trainLabels.npy'))



    # # upsample the small Tensors:
    # train_jin_acc = torch.Tensor(np.repeat(train_jin_acc, 10, axis=1))
    # train_jin_gyro = torch.Tensor(np.repeat(train_jin_gyro, 10, axis=1))
    # train_mag_meter = torch.Tensor(np.repeat(train_mag_meter, 4, axis=1))

    # train_ms_acc = torch.Tensor(zoom(train_ms_acc, (1, 200/67, 1)))
    # train_ms_gyro = torch.Tensor(zoom(train_ms_gyro, (1, 200/67, 1)))


    # concat all training Tensors:
    train_ls = [train_acc_meter, train_grav, train_gyro, train_jin_acc, train_jin_gyro, train_lin_acc, train_mag_meter,
                train_ms_acc, train_ms_gyro]

    # train_tens = torch.cat(train_ls, dim=2)

    """
    change if activate when 2D or 1D
    """
    # train_tens = train_tens.view(2284,1, 27, 800)
    # train_ls = train_tens.view(-1, 3, 800)

    # print(train_tens.shape)
    if cuda:
        return [DataLoader(TensorDataset(sensor.cuda(), torch.LongTensor(train_labels.long()).cuda()), batch_size=32, shuffle=True) for sensor in train_ls]
    else:
        return [DataLoader(TensorDataset(sensor, torch.LongTensor(train_labels.long())), batch_size=32, shuffle=True) for sensor in train_ls]


def get_simple_eval_loader(cuda=False):
    eval_acc_meter = torch.Tensor(np.load('../../dataset/testing/testAccelerometer.npy'))
    eval_grav = torch.Tensor(np.load('../../dataset/testing/testGravity.npy'))
    eval_gyro = torch.Tensor(np.load('../../dataset/testing/testGyroscope.npy'))
    eval_jin_acc = np.load('../../dataset/testing/testJinsAccelerometer.npy')
    eval_jin_gyro = np.load('../../dataset/testing/testJinsGyroscope.npy')
    eval_lin_acc = torch.Tensor(np.load('../../dataset/testing/testLinearAcceleration.npy'))
    eval_mag_meter = np.load('../../dataset/testing/testMagnetometer.npy')
    eval_ms_acc = np.load('../../dataset/testing/testMSAccelerometer.npy')
    eval_ms_gyro = np.load('../../dataset/testing/testMSGyroscope.npy')

    eval_labels = torch.Tensor(np.load('../../dataset/testing/testLabels.npy'))

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

    """
    change between 2D and 1D
    """
    eval_tens = eval_tens.view(-1, 1, 27, 800)
    #eval_tens = eval_tens.view(-1, 27, 800)

    # print(train_tens.shape)

    if cuda:
        data_set_eval = TensorDataset(eval_tens.cuda(), torch.LongTensor(eval_labels.long()).cuda())
    else:
        data_set_eval = TensorDataset(eval_tens, torch.LongTensor(eval_labels.long()))

    # print(data_set_train[0])

    data_loader_eval = DataLoader(data_set_eval, batch_size=1, shuffle=False)

    return data_loader_eval


if __name__ == "__main__":

    test_dataloader = get_simple_data_loader()