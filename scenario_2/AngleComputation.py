import numpy as np


class AngleComputation:

    def __init__(self, acc_data,  len_init_time=9, freq=400):
        self.g_b_dash_vec = self.update_g_b_dash(acc_data, len_init_time, freq)


    def get_init_g_b_dash(self, acc_data, len_init_time = 9, freq = 400):

        # presumably smooth the data before averaging it...

        g_x = np.mean(acc_data[:len_init_time * freq, 1])
        g_y = np.mean(acc_data[:len_init_time * freq, 2])
        g_z = np.mean(acc_data[:len_init_time * freq, 3])

        gb_dash = [g_x, g_y, g_z]
        return gb_dash


    def update_g_b_dash(self, acc_data,  len_init_time=9, freq=400):
        mu = 0.9    # perhaps something else is better [0.5,1[
        g_b_dash_init = self.get_init_g_b_dash(acc_data, len_init_time, freq)

        g_b_dash_up = np.zeros([len(acc_data), 3])
        # the 10 sec phase at the beginning that is used to compute the init_g_b_dash is fillt with the init_g_b_dash <-coorect?
        g_b_dash_up[:len_init_time * freq, :] = g_b_dash_init

        for i in range(len_init_time * freq, len(acc_data)):
            g_b_dash_up[i] = mu * g_b_dash_up[i-1] + (1-mu) * acc_data[i, 1:]

        return g_b_dash_up


    def get_R(self, g_b_dash):

        gb = [0, 0, 9.81]   #not needed?

        u_z = g_b_dash
        u_x = np.cross([0, 1, 0], u_z)
        u_y = np.cross(u_z, u_x)

        u_z_norm = np.linalg.norm(u_z)
        u_x_norm = np.linalg.norm(u_x)
        u_y_norm = np.linalg.norm(u_y)

        R = np.array([u_x/u_x_norm, u_y/u_y_norm, u_z/u_z_norm])

        return R

    def get_theta_dot(self, g_b_dash, omega):
        R = self.get_R(g_b_dash)
        R_inv = R.transpose()

        theta_dot = R_inv.dot(omega)

        return theta_dot

    def get_angle(self, omega, step_idxs, delta_t):

        angles = np.zeros([len(step_idxs)])

        for enum, step_idx in enumerate(step_idxs):

            sum_theta_t = sum(self.get_theta_dot(self.g_b_dash_vec[step_idx], omega[step_idx, 1:]) * delta_t)

            angles[enum] = sum_theta_t

        return angles
