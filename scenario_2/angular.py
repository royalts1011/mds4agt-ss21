import numpy as np
from scipy import linalg


def calc_rotation_matrix(g_b_dash):
    '''
    calculate rotation matrix for given g_b_dash
    :param g_b_dash:
    :return: Rotation matrix with respecting the transformation from g_b_dash to g_b
    '''
    foo = np.array([0, 1, 0])

    u_z = g_b_dash
    u_x = np.cross(foo, u_z)
    u_y = np.cross(u_z, u_x)

    r = np.stack([u_x/linalg.norm(u_x), u_y/linalg.norm(u_y), u_z/linalg.norm(u_z)], axis=1)
    return r


def calc_initial_g_b_dash(acc_data, len_init_time=9, freq=400):
    '''
    calculate the initial g_b_dash
    :param acc_data: should be preprocessed acc_data at this point
    :param len_init_time: time where the smartphone os not moved at the beginning
    :param freq: sampling rate
    :return: inital g_b_dash for averaged acc data
    '''
    g_x = np.mean(acc_data[:len_init_time * freq, 0])
    g_y = np.mean(acc_data[:len_init_time * freq, 1])
    g_z = np.mean(acc_data[:len_init_time * freq, 2])

    initial_g_b_dash = np.array([g_x, g_y, g_z])
    return initial_g_b_dash


def calc_all_g_b_dash(acc_data,initial_g_b_dash, m):
    """
    compute all g_b_dash for angular
    if: t = 1 ---> initial_gb_dash
    else: m * all_g_b_dash[t - 1] + (1 - m) * acc_data[t]

    return: all needed g_b_dash vectors stored in one Numpy Array
    """
    all_g_b_dash = np.zeros_like(acc_data)
    all_g_b_dash[0] = initial_g_b_dash  # for a better init
    for t in range(1, len(acc_data)):
        all_g_b_dash[t] = m * all_g_b_dash[t - 1] + (1 - m) * acc_data[t]
    return all_g_b_dash
