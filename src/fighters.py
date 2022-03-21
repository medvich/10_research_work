import numpy as np
from math import *
from ambiance import Atmosphere
from copy import deepcopy
from easyvec import Vec3


def get_grav_accel(altitude):
    return float(Atmosphere(altitude).grav_accel)


def get_density(altitude):
    return float(Atmosphere(altitude).density)


def get_speed_of_sound(altitude):
    return float(Atmosphere(altitude).speed_of_sound)


def get_spatial_angle(alpha, beta, **kwargs):
    res = atan(sqrt(tan(alpha) ** 2 + tan(beta) ** 2))
    if 'key' not in kwargs or kwargs['key'] == 'rad':
        return res
    if kwargs['key'] == 'deg':
        return np.rad2deg(res)
    raise Exception('Invalid key')


def get_plane_angles(spatial_angle, alpha, beta, **kwargs):
    r = tan(spatial_angle)
    phi = atan(tan(alpha) / tan(beta)) if abs(beta) > 1e-9 else pi / 2
    alpha_cor = copysign(atan(r * sin(phi)), alpha)
    beta_cor = copysign(atan(r * cos(phi)), beta)
    if 'key' not in kwargs or kwargs['key'] == 'rad':
        return alpha_cor, beta_cor
    if kwargs['key'] == 'deg':
        return np.rad2deg(alpha_cor), np.rad2deg(beta_cor)
    raise Exception('Invalid key')


def get_angle_to(r, vel):
    """
    :param r: numpy array([r, φ, χ])
    :param vel: numpy array([υ, Θ, Ψ])
    :return: angle(r,v) (rad)
    """
    r, phi, chi = r
    vel, theta, psi = vel
    r_coordinates = np.array([r*cos(phi)*cos(chi),
                              r*sin(phi),
                              r*cos(phi)*sin(chi)])
    vel_coordinates = np.array([vel*cos(theta)*cos(psi),
                                vel*sin(theta),
                                vel*cos(theta)*sin(psi)])
    r_vec = Vec3.from_list(r_coordinates)
    vel_vec = Vec3.from_list(vel_coordinates)
    return abs(r_vec.angle_to(vel_vec))


class Missile:

    def __init__(self, opts):
        """
        Для инициализации объекта класса Missile необходимо
        на вход послать словарь следующей структуры:
        opts =
        {
            'initial_state': {
                'x0': int/float (м),
                'y0': int/float (м),
                'z0': int/float (м),
                'vel0': int/float (м/с),
                'theta0': int/float (рад),
                'psi0': int/float (рад) },

            'energetics': {
                'mass0': int/float (кг),
                'omega_act': int/float (кг),
                'omega_march': int/float (кг),
                't_act': int/float (с),
                't_march': int/float (с),
                'I_1': int/float (м/с) },

            'aerodynamics': {
                'D': int/float (м),
                'spatial_angle_max': int/float (рад),
                'Ms': numpy array,
                'Cx0_arr': numpy array,
                'Cya_arr': numpy array,
                'Cyb_arr': numpy array }
        }
        """
        self.initial_state = np.array([opts['initial_state']['x0'],
                                       opts['initial_state']['y0'],
                                       opts['initial_state']['z0'],
                                       opts['initial_state']['vel0'],
                                       opts['initial_state']['theta0'],
                                       opts['initial_state']['psi0']
                                       ])
        self.aerodynamics = deepcopy(opts['aerodynamics'])
        self.Sm = pi * self.aerodynamics['D'] ** 2 / 4
        self.mass0 = opts['energetics']['mass0']
        self.omega_act = opts['energetics']['omega_act']
        self.omega_march = opts['energetics']['omega_march']
        self.t_act = opts['energetics']['t_act']
        self.t_march = opts['energetics']['t_march']
        self.I_1 = opts['energetics']['I_1']
        self.P_act = self.omega_act / self.t_act * self.I_1
        if self.t_march == 0:
            self.P_march = 0
        else:
            self.P_march = self.omega_march / self.t_march * self.I_1
        self.ts = np.array([0, self.t_act, self.t_act + self.t_march])
        self.omegas = np.array([0, self.omega_act, self.omega_act + self.omega_march])
        self.Ps = np.array([-1, self.P_act, self.P_march, -1])
        self.initial_params = np.array([self.mass0,
                                        self.P_act,
                                        self.get_Cya(opts['initial_state']['vel0']),
                                        get_density(opts['initial_state']['y0']),
                                        get_grav_accel(opts['initial_state']['y0']),
                                        get_speed_of_sound(opts['initial_state']['y0'])
                                        ])
        self.log = []
        self.current_state = self.initial_state
        self.current_params = self.initial_params

    def get_current_state(self):
        return self.current_state

    def reset(self):
        self.log.clear()
        self.current_state = self.initial_state
        self.current_params = self.initial_params
        return self.initial_state

    def get_ode_solution(self, t, state, alpha, beta):
        x, y, z, vel, theta, psi = state
        spatial_angle = get_spatial_angle(alpha, beta)
        if spatial_angle > self.aerodynamics['spatial_angle_max']:
            spatial_angle = self.aerodynamics['spatial_angle_max']
        alpha, beta = get_plane_angles(spatial_angle, alpha, beta)
        P = max(self.get_P(t), 0)
        omega = self.get_omega(t)
        m = self.mass0 - omega
        g = get_grav_accel(y)
        rho = get_density(y)
        a = get_speed_of_sound(y)
        M = vel / a
        Cya = self.get_Cya(M)
        Cyb = self.get_Cyb(M)
        Cx0 = self.get_Cx0(M)
        Cx = Cx0 + abs(Cya * np.rad2deg(alpha) * tan(alpha)) + abs(Cyb * np.rad2deg(beta) * tan(beta))
        x_force = Cx * rho * vel ** 2 / 2 * self.Sm
        y_force = Cya * np.rad2deg(alpha) * rho * vel ** 2 / 2 * self.Sm * sqrt(2)
        z_force = Cyb * np.rad2deg(beta) * rho * vel ** 2 / 2 * self.Sm * sqrt(2)
        self.log.append((x, y, z, vel, theta, psi, alpha, beta, spatial_angle))
        self.current_state = np.array([x, y, z, vel, theta, psi])
        self.current_params = np.array([m, P, Cya, rho, g, a])
        return np.array([
            vel * cos(theta) * cos(psi),
            vel * sin(theta),
            vel * cos(theta) * sin(psi),
            (P * cos(alpha) * cos(beta) - x_force) / m - g * sin(theta),
            ((y_force + P * sin(alpha)) / (m * g) - cos(theta)) * g / vel,
            (P * cos(alpha) * sin(beta) - z_force) / m / vel / cos(theta)
        ], copy=False)

    def get_omega(self, t):
        return np.interp(t, self.ts, self.omegas)

    def get_P(self, t):
        i = np.searchsorted(self.ts, t, side='right')
        return self.Ps[i]

    def get_Cx0(self, M):
        return np.interp(M, self.aerodynamics['Ms'], self.aerodynamics['Cx0_arr'])

    def get_Cya(self, M):
        return np.interp(M, self.aerodynamics['Ms'], self.aerodynamics['Cya_arr'])

    def get_Cyb(self, M):
        return np.interp(M, self.aerodynamics['Ms'], self.aerodynamics['Cyb_arr'])

    def get_required_alpha(self, delta_theta):
        x, y, z, vel, theta, psi = self.current_state
        m, P, Cya, rho, g, a = self.current_params
        return np.radians((vel / g * delta_theta + cos(theta)) * m * g
                          / (P / 57.3 + Cya * rho * vel ** 2 / 2 * self.Sm * sqrt(2)))

    def get_required_beta(self, delta_psi, required_alpha):
        x, y, z, vel, theta, psi = self.current_state
        m, P, Cya, rho, g, a = self.current_params
        return np.radians(-m * vel * cos(theta) * delta_psi
                          / (-P / 57.3 * cos(required_alpha) + Cya * rho * vel ** 2 / 2 * self.Sm * sqrt(2)))


class Target:

    MASS = 1000
    THRUST = 10000
    D = 2000e-3

    def __init__(self, opts):
        """
        Для инициализации объекта класса Target необходимо
        на вход послать словарь следующей структуры:
        opts =
        {
            'x0': int/float (м),
            'y0': int/float (м),
            'z0': int/float (м),
            'vel0': int/float (м/с),
            'theta0': int/float (рад),
            'psi0': int/float (рад)
        }
        Энергетические параметры принимаются постоянными,
        запас топлива и время работы двигателя не ограничены.
        """
        self.initial_state = np.array([opts['x0'],
                                       opts['y0'],
                                       opts['z0'],
                                       opts['vel0'],
                                       opts['theta0'],
                                       opts['psi0']
                                       ])
        self._init_coefficients()
        self.Sm = pi * self.D ** 2 / 4
        self.log = []
        self.current_state = self.initial_state

    def get_current_state(self):
        return self.current_state

    def reset(self):
        self.log.clear()
        self.current_state = self.initial_state
        return self.initial_state

    def get_ode_solution(self, state, alpha, beta):
        x, y, z, vel, theta, psi = state
        P = self.THRUST
        m = self.MASS
        g = get_grav_accel(y)
        rho = get_density(y)
        a = get_speed_of_sound(y)
        M = vel / a
        Cya = self.get_Cya(M)
        Cyb = self.get_Cyb(M)
        Cx0 = self.get_Cx0(M)
        Cx = Cx0 + abs(Cya * np.rad2deg(alpha) * tan(alpha)) + abs(Cyb * np.rad2deg(beta) * tan(beta))
        x_force = Cx * rho * vel ** 2 / 2 * self.Sm
        y_force = Cya * np.rad2deg(alpha) * rho * vel ** 2 / 2 * self.Sm * sqrt(2)
        z_force = Cyb * np.rad2deg(beta) * rho * vel ** 2 / 2 * self.Sm * sqrt(2)
        self.log.append((x, y, z, vel, theta, psi, alpha, beta))
        self.current_state = np.array([x, y, z, vel, theta, psi])
        return np.array([
            vel * cos(theta) * cos(psi),
            vel * sin(theta),
            vel * cos(theta) * sin(psi),
            (P * cos(alpha) * cos(beta) - x_force) / m - g * sin(theta),
            ((y_force + P * sin(alpha)) / (m * g) - cos(theta)) * g / vel,
            (P * cos(alpha) * sin(beta) - z_force) / m / vel / cos(theta)
        ], copy=False)

    def _init_coefficients(self):
        self.M_arr = np.array([0.1, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05,
                               1.1, 1.15, 1.5, 1.9, 2.3, 2.7, 3.1, 3.5, 3.9, 4.3, 4.7])
        self.Cx0_arr = np.array([0.2958, 0.2998, 0.3078, 0.3198, 0.3234, 0.3273, 0.3314,
                                 0.3358, 0.3404, 0.3453, 0.3504, 0.3558, 0.3614, 0.4426,
                                 0.3598, 0.2885, 0.2376, 0.2009, 0.1741, 0.1533, 0.1377,
                                 0.1255]) * 0.5
        self.Cya_arr = np.array([0.0571, 0.0577, 0.0591, 0.0615, 0.0623, 0.0633, 0.0643,
                                 0.0655, 0.0671, 0.0708, 0.0794, 0.0814, 0.0802, 0.0664,
                                 0.0543, 0.0465, 0.042, 0.0387, 0.0358, 0.0331, 0.0312,
                                 0.0291]) * 0.5
        self.Cyb_arr = self.Cya_arr

    def get_Cx0(self, M):
        return np.interp(M, self.M_arr, self.Cx0_arr)

    def get_Cya(self, M):
        return np.interp(M, self.M_arr, self.Cya_arr)

    def get_Cyb(self, M):
        return np.interp(M, self.M_arr, self.Cyb_arr)


class LineOfSight:

    def __init__(self, missile, target):
        """
        Для инициаоизации объекта класса LineOfSight (по-человечески,
        это линия визирования) необходимо на вход послать numpy массивы
        следующей структуры:
        initial_state: numpy array([r0, φ0, χ0])
        """
        r_coordinates = np.array([target.get_current_state()[0] - missile.get_current_state()[0],
                                  target.get_current_state()[1] - missile.get_current_state()[1],
                                  target.get_current_state()[2] - missile.get_current_state()[2]])
        r_coordinates_proj = np.where(r_coordinates == r_coordinates[1], 0, r_coordinates)
        r = Vec3.from_list(r_coordinates)
        r_proj = Vec3.from_list(r_coordinates_proj)
        i = Vec3(1, 0, 0)
        phi = r_proj.angle_to(r)
        chi = pi - i.angle_to(r_proj)
        self.initial_state = np.array([r.len(), phi, chi])
        self.current_state = self.initial_state
        self.log = []

    def reset(self):
        self.log.clear()
        return self.initial_state

    def get_los_ode_solution(self, state, missile_v_vec, target_v_vec):
        """
        :param state: numpy array([r, φ, χ])
        :param missile_v_vec: numpy array([υ, Θ, Ψ])
        :param target_v_vec: numpy array([υ_c, Θ_c, Ψ_c])
        :return: numpy array([dr/dt, dφ/dt, dχ/dt])
        """
        missile_vel, missile_theta, missile_psi = missile_v_vec
        target_vel, target_theta, target_psi = target_v_vec
        r, phi, chi = state
        coefficient = np.array([1, r, r * cos(phi)])
        target_t = np.array([cos(target_theta) * cos(target_psi - chi) * cos(phi),
                             -cos(target_theta) * cos(target_psi - chi) * sin(phi),
                             cos(target_theta) * sin(target_psi - chi)])
        target_n = np.array([sin(target_theta) * sin(phi),
                             sin(target_theta) * cos(phi),
                             0])
        missile_t = np.array([cos(missile_theta) * cos(missile_psi - chi) * cos(phi),
                              -cos(missile_theta) * cos(missile_psi - chi) * sin(phi),
                              cos(missile_theta) * sin(missile_psi - chi)])
        missile_n = np.array([sin(missile_theta) * sin(phi),
                              sin(missile_theta) * cos(phi),
                              0])
        self.log.append((r, phi, chi))
        self.current_state = np.array([r, phi, chi])
        return 1 / coefficient * (target_vel * (target_t + target_n) - missile_vel * (missile_t + missile_n))


def simulation(missile_opts, target_opts, target_autopilot, pn_guidance_law, k, t=0, dt=0.01, t_integration=1000):
    t_stats = []
    v_angle_to_r_stats = []
    missile = Missile(missile_opts)
    target = Target(target_opts)
    los = LineOfSight(missile, target)
    y_missile = missile.reset()
    y_target = target.reset()
    y_los = los.reset()
    max_iter = int(t_integration / dt)
    for i in range(max_iter):
        if y_missile[1] < 0:
            print('Мы упали')
            break
        if y_missile[1] > 70000:
            print('Мы улетели слишком высоко')
            break
        if y_target[1] < 0:
            print('Цель упала')
            break
        if y_target[1] > 70000:
            print('Цель улетела слишком высоко')
            break
        delta_los = -los.get_los_ode_solution(y_los, y_missile[-3:], y_target[-3:])
        delta_theta, delta_psi = pn_guidance_law(k, delta_los[1], delta_los[2])
        missile_alpha = missile.get_required_alpha(delta_theta)
        missile_beta = missile.get_required_beta(delta_psi, missile_alpha)
        target_alpha, target_beta = target_autopilot()
        y1_target = y_target + dt * target.get_ode_solution(y_target, target_alpha, target_beta)
        y1_missile = y_missile + dt * missile.get_ode_solution(t, y_missile, missile_alpha, missile_beta)
        y1_los = y_los + dt * delta_los
        v_angle_to_r = pi - get_angle_to(y_los, y_missile[-3:])
        t_stats.append(t)
        v_angle_to_r_stats.append(v_angle_to_r)
        t += dt
        if y_los[0] < 30:
            print('Мы попали')
            break
        if 180 / pi * v_angle_to_r > 30:
            print('Мы потеряли цель из вида')
            break
        if y_missile[3] < 200:
            print('Мы замедлились и потеряли управление')
            break
        if y_target[3] < 200:
            print('Цель замедлилась и потеряла управление')
            break
        y_missile = y1_missile
        y_target = y1_target
        y_los = y1_los
    else:
        print('Мы совершили максимальное число шагов интегрирования')
    missile_stats = np.array(missile.log)
    target_stats = np.array(target.log)
    los_stats = np.array(los.log)
    return {
        't': t_stats,
        'r': los_stats[:, 0],
        'phi': los_stats[:, 1],
        'chi': los_stats[:, 2],
        'missile_x': missile_stats[:, 0],
        'missile_y': missile_stats[:, 1],
        'missile_z': missile_stats[:, 2],
        'missile_vel': missile_stats[:, 3],
        'missile_theta': missile_stats[:, 4],
        'missile_psi': missile_stats[:, 5],
        'missile_alpha': missile_stats[:, 6],
        'missile_beta': missile_stats[:, 7],
        'missile_spatial_angle': missile_stats[:, 8],
        'target_x': target_stats[:, 0],
        'target_y': target_stats[:, 1],
        'target_z': target_stats[:, 2],
        'target_vel': target_stats[:, 3],
        'target_theta': target_stats[:, 4],
        'target_psi': target_stats[:, 5]
    }
