import numpy as np
from math import *
from ambiance import Atmosphere
from copy import deepcopy
from easyvec import Vec2
from gym import Env
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt
from matplotlib.pylab import figure
from IPython.display import clear_output


def get_grav_accel(altitude):
    return float(Atmosphere(altitude).grav_accel)


def get_density(altitude):
    return float(Atmosphere(altitude).density)


def get_speed_of_sound(altitude):
    return float(Atmosphere(altitude).speed_of_sound)


def parametric_sigmoid(alpha, beta, x):
    """
    :param alpha: current velocity
    :param beta: time step 'tau' or 'dt'
    :param x: dr/dt
    """
    if x <= 0:
        return 0
    elif 0 < x < alpha*beta:
        return x / alpha / beta
    else:
        return 1


def normal_distribution(sigma, mu, x):
    return 1 / sigma / sqrt(2*pi) * np.exp(-(x - mu)**2 / (2*sigma**2))


def weights_foo(distance, max_distance, n):
    return (abs(distance) / max_distance)**n


class Missile:
    ALPHA = 4  # град
    BETA_MAX = 25  # град
    I_1 = 1500  # м/с
    D = 160e-3  # м
    MASS_WARHEAD = 25  # кг
    MASS_SEEKER = 12  # кг
    MASS_STEERING = 10  # кг
    ENGINE_PERFECTION_COEFFICIENT = 1.3  # кг
    MIN_VELOCITY = 310  # м/с

    def __init__(self, opts, altitude):
        self.state = np.array([opts['initial_state']['x0'],
                               opts['initial_state']['z0'],
                               opts['initial_state']['vel0'],
                               opts['initial_state']['psi0']
                               ], dtype=np.float32)
        self.aerodynamics = deepcopy(opts['aerodynamics'])
        self.Sm = pi * self.D ** 2 / 4
        self.omega_act = opts['energetics']['omega_act']
        self.omega_march = opts['energetics']['omega_march']
        self.t_act = opts['energetics']['t_act']
        self.t_march = opts['energetics']['t_march']
        self.mass0 = self.MASS_WARHEAD + self.MASS_SEEKER + self.MASS_STEERING + self.ENGINE_PERFECTION_COEFFICIENT \
            * (self.omega_act + self.omega_march)
        self.thrust_act = self.omega_act / self.t_act * self.I_1
        if self.t_march == 0.0:
            self.thrust_march = 0.0
        else:
            self.thrust_march = self.omega_march / self.t_march * self.I_1
        self.ts = np.array([0, self.t_act, self.t_act + self.t_march])
        self.omegas = np.array([0, self.omega_act, self.omega_act + self.omega_march])
        self.thrusts = np.array([-1, self.thrust_act, self.thrust_march, -1])

        # Высота полета (постоянная)
        self.altitude = altitude

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_ds(self, t, beta):
        x, z, vel, psi = self.get_state()
        thrust, m, omega, g, rho, a, mach, cya, cx0 = self.get_params(t)
        alpha = np.radians(self.ALPHA)
        cyb = cya
        cx = cx0 + abs(cya * np.rad2deg(alpha) * tan(alpha)) + abs(cyb * np.rad2deg(beta) * tan(beta))
        x_force = cx * rho * vel ** 2 / 2 * self.Sm
        z_force = -cya * np.rad2deg(beta) * rho * vel ** 2 / 2 * self.Sm * sqrt(2)
        self.state = np.array([x, z, vel, psi], dtype=np.float32)
        return np.array([
            vel * cos(psi),
            vel * sin(psi),
            (thrust * cos(alpha) * cos(beta) - x_force) / m,
            (thrust * cos(alpha) * sin(beta) - z_force) / m / vel
        ], copy=False, dtype=np.float32)

    def get_omega(self, t):
        return np.interp(t, self.ts, self.omegas)

    def get_thrust(self, t):
        i = np.searchsorted(self.ts, t, side='right')
        return self.thrusts[i]

    def get_cx0(self, mach):
        return np.interp(mach, self.aerodynamics['Ms'], self.aerodynamics['Cx0_arr'])

    def get_cya(self, mach):
        return np.interp(mach, self.aerodynamics['Ms'], self.aerodynamics['Cya_arr'])

    def get_params(self, t):
        x, z, vel, psi = self.get_state()
        thrust = max(self.get_thrust(t), 0)
        omega = self.get_omega(t)
        m = self.mass0 - omega
        g = get_grav_accel(self.altitude)
        rho = get_density(self.altitude)
        a = get_speed_of_sound(self.altitude)
        mach = vel / a
        cya = self.get_cya(mach)
        cx0 = self.get_cx0(mach)
        return np.array([thrust, m, omega, g, rho, a, mach, cya, cx0], copy=False, dtype=np.float32)

    def get_required_beta(self, t, dpsi):
        x, z, vel, psi = self.get_state()
        thrust, m, omega, g, rho, a, mach, cya, cx0 = self.get_params(t)
        required_beta = (-m * vel * dpsi) / (-thrust / 57.3 * cos(np.radians(self.ALPHA))
                                             + cya * rho * vel ** 2 / 2 * self.Sm * sqrt(2))
        return np.radians(required_beta)


class Target:
    ALPHA = 4  # град
    MASS = 700  # кг
    THRUST = 11000  # Н
    D = 1500e-3  # м

    def __init__(self, opts, altitude):
        self.state = np.array([opts['x0'],
                               opts['z0'],
                               opts['vel0'],
                               opts['psi0']
                               ], dtype=np.float32)
        self._init_coefficients()
        self.Sm = pi * self.D ** 2 / 4

        # Высота полета (постоянная)
        self.altitude = altitude

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_ds(self, beta):
        x, z, vel, psi = self.get_state()
        alpha = np.radians(self.ALPHA)
        m = self.MASS
        thrust = self.THRUST
        rho = get_density(self.altitude)
        a = get_speed_of_sound(self.altitude)
        mach = vel / a
        cya = self.get_cya(mach)
        cx0 = self.get_cx0(mach)
        # cx = cx0 + abs(cya * np.rad2deg(alpha) * tan(alpha)) + abs(cyb * np.rad2deg(beta) * tan(beta)) - прав. вырж-е
        cx = cx0 + abs(cya * np.rad2deg(alpha) * tan(alpha))
        x_force = cx * rho * vel ** 2 / 2 * self.Sm
        z_force = -cya * np.rad2deg(beta) * rho * vel ** 2 / 2 * self.Sm * sqrt(2)
        self.state = np.array([x, z, vel, psi], dtype=np.float32)
        return np.array([
            vel * cos(psi),
            vel * sin(psi),
            # (thrust * cos(alpha) * cos(beta) - x_force) / m - правильное выражение
            (thrust * cos(alpha) - x_force) / m,
            (thrust * cos(alpha) * sin(beta) - z_force) / m / vel
        ], copy=False, dtype=np.float32)

    def _init_coefficients(self):
        self.mach_arr = np.array([0.1, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05,
                                  1.1, 1.15, 1.5, 1.9, 2.3, 2.7, 3.1, 3.5, 3.9, 4.3, 4.7])
        self.cx0_arr = np.array([0.2958, 0.2998, 0.3078, 0.3198, 0.3234, 0.3273, 0.3314,
                                 0.3358, 0.3404, 0.3453, 0.3504, 0.3558, 0.3614, 0.4426,
                                 0.3598, 0.2885, 0.2376, 0.2009, 0.1741, 0.1533, 0.1377,
                                 0.1255])
        self.cya_arr = np.array([0.0571, 0.0577, 0.0591, 0.0615, 0.0623, 0.0633, 0.0643,
                                 0.0655, 0.0671, 0.0708, 0.0794, 0.0814, 0.0802, 0.0664,
                                 0.0543, 0.0465, 0.042, 0.0387, 0.0358, 0.0331, 0.0312,
                                 0.0291]) * 0.9

    def get_cx0(self, mach):
        return np.interp(mach, self.mach_arr, self.cx0_arr)

    def get_cya(self, mach):
        return np.interp(mach, self.mach_arr, self.cya_arr)


class LineOfSight:

    def __init__(self, missile_state, target_state):
        r_coordinates = np.array([target_state[0] - missile_state[0],
                                  target_state[1] - missile_state[1]], dtype=np.float32)
        r = Vec2.from_list(r_coordinates)
        i = Vec2(1, 0)
        chi = pi - i.angle_to(r)
        self.state = np.array([r.len(), chi], dtype=np.float32)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_ds(self, missile_state, target_state):
        r, chi = self.get_state()
        missile_x, missile_z, missile_vel, missile_psi = missile_state
        target_x, target_z, target_vel, target_psi = target_state
        coefficient = np.array([1.0, r], dtype=np.float32)
        target_t = np.array([cos(target_psi - chi), sin(target_psi - chi)], dtype=np.float32)
        target_n = np.array([0.0, 0.0])
        missile_t = np.array([cos(missile_psi - chi), sin(missile_psi - chi)], dtype=np.float32)
        missile_n = np.array([0.0, 0.0])
        return 1 / coefficient * (target_vel * (target_t + target_n) - missile_vel * (missile_t + missile_n))


class AirCombat2D(Env):
    MISSILE_ETTA_MAX = 25.0  # град
    DETECTION_DISTANCE_MAX = 60e3  # м
    EXPLOSION_DISTANCE = 10.0  # м
    TERMINATION_TIME = 100.0  # с

    K = 1.7  # это коэффициент используется в формуле для опрделения реварда по расстоянию
    # (по какой-то неведомой причине текущий dr этом реварде очень часто переваливает за верхнюю границу сигмоиды,
    # хотя та должна строится по максимально возможному dr в текущей ситуации (исходя из текущей скорости ракеты))

    def __init__(self, opts, **kwargs):
        # Инициализируем и обрабатываем входные данные
        self.opts = deepcopy(opts)
        if 'x0' not in self.opts['target']:
            self.opts['target']['x0'] = 0.0
        if 'z0' not in self.opts['target']:
            self.opts['target']['z0'] = 0.0
        if ('x0' in self.opts['missile']['initial_state']) or ('z0' in self.opts['missile']['initial_state']):
            raise Exception("Missile coordinates must fill from LOS options")
        self.opts['missile']['initial_state']['x0'] = self.opts['los']['r0'] * cos(self.opts['los']['chi0'])
        self.opts['missile']['initial_state']['z0'] = self.opts['los']['r0'] * sin(self.opts['los']['chi0'])
        if 't_march' not in self.opts['missile']['energetics']:
            self.opts['missile']['energetics']['t_march'] = 0.0
        if 'omega_march' not in self.opts['missile']['energetics']:
            self.opts['missile']['energetics']['omega_march'] = 0.0
        if ('Ms' not in self.opts['missile']['aerodynamics']) or \
                ('Cx0_arr' not in self.opts['missile']['aerodynamics']) or \
                ('Cya_arr' not in self.opts['missile']['aerodynamics']):
            raise Exception("Missile aerodynamic characteristics must be filled")

        # Инициализация объектов исследования:
        # altitude - высота полета
        altitude = kwargs['altitude'] if 'altitude' in kwargs else 15e3
        self.target = Target(self.opts['target'], altitude)
        self.missile = Missile(self.opts['missile'], altitude)
        self.los = LineOfSight(self.missile.get_state(), self.target.get_state())

        # Функция управления цели
        assert 'target_autopilot' in kwargs, 'Need to add target autopilot function'
        self.target_autopilot = kwargs['target_autopilot']

        # Временные параметры моделирования
        self.tau = kwargs['dt'] if 'dt' in kwargs else 0.05
        self.t0 = kwargs['t0'] if 't0' in kwargs else 0.0

        # Набор возможных действий ракеты
        self.ACTIONS = [-1.0, 0.0, 1.0]

        # Размерность пространства возможных действий
        self.action_space = Discrete(len(self.ACTIONS))

        # Максимальные и минимальные наблюдаемые ракетой параметры состояния среды
        # Наблюдаемые параметры: r, χ, η, β, υ, Ψ, υ_c, Ψ_c
        high = np.array(
            [20e3,
             2 * pi,
             2 * pi,
             np.radians(self.missile.BETA_MAX),
             2000.0,
             2 * pi,
             2000.0,
             2 * pi,
             135e2], dtype=np.float32
        )
        low = np.array(
            [-2e3,
             -2 * pi,
             -2 * pi,
             -np.radians(self.missile.BETA_MAX),
             0.0,
             -2 * pi,
             0.0,
             -2 * pi,
             0], dtype=np.float32
        )

        # Задаем пространство состояний
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        # Текущее состояние среды
        self.state = None

        # Текущие значения параметров
        self.t = None
        self.missile_beta = None
        self.missile_etta = None
        self.target_etta = None
        self.target_beta = None
        self.log = None
        self.reward_data = None

    def reset(self, **kwargs):
        self.t = self.t0
        self.missile_beta = 0.0
        self.missile_etta = self.get_etta(self.missile.get_state(), self.los.get_state())
        self.target_etta = self.get_etta(self.target.get_state(), self.los.get_state() * np.array([-1, 1]))

        self.state = np.array(
            [self.opts['los']['r0'],
             self.opts['los']['chi0'],
             self.missile_etta,
             self.missile_beta,
             self.opts['missile']['initial_state']['vel0'],
             self.opts['missile']['initial_state']['psi0'],
             self.opts['target']['vel0'],
             self.opts['target']['psi0'],
             self.missile.get_thrust(self.t)], dtype=np.float32
        )
        reward0 = 0.0
        log0 = np.concatenate([
            np.array(
                [self.t, self.missile_etta, self.target_etta, self.missile_beta, reward0], dtype=np.float32
            ),
            self.los.get_state(),
            self.missile.get_state(),
            self.target.get_state()
        ])
        self.log = np.array([log0], dtype=np.float32)

        return self.state

    @staticmethod
    def get_etta(actor_state, los_state):
        _, _, vel, psi = actor_state
        r, chi = los_state
        r_coordinates = np.array([r * cos(chi),
                                  r * sin(chi)], dtype=np.float32)
        vel_coordinates = np.array([vel * cos(psi),
                                    vel * sin(psi)], dtype=np.float32)
        r_vec = -Vec2.from_list(r_coordinates)
        vel_vec = Vec2.from_list(vel_coordinates)
        return r_vec.angle_to(vel_vec)

    def step(self, action):
        s = self.state
        assert s is not None, "Call reset before using AirCombat2DEnv object."

        missile_state = self.missile.get_state()
        target_state = self.target.get_state()
        los_state = self.los.get_state()

        distance = los_state[0]

        self.target_beta = self.target_autopilot(self.t)
        self.missile_beta = self.missile_beta + np.radians(self.ACTIONS[action])

        if self.missile_beta > np.radians(self.missile.BETA_MAX):
            self.missile_beta = np.radians(self.missile.BETA_MAX)
        if self.missile_beta < -np.radians(self.missile.BETA_MAX):
            self.missile_beta = -np.radians(self.missile.BETA_MAX)

        missile_ds = self.missile.get_ds(self.t, self.missile_beta)
        target_ds = self.target.get_ds(self.target_beta)
        los_ds = self.los.get_ds(missile_state, target_state)
        missile_state = missile_state + self.tau * missile_ds
        target_state = target_state + self.tau * target_ds
        los_state = los_state - self.tau * los_ds

        next_distance = los_state[0]

        self.missile_etta = self.get_etta(missile_state, los_state)
        self.target_etta = self.get_etta(target_state, los_state * np.array([-1, 1]))

        missile_thrust = self.missile.get_thrust(self.t)

        self.missile.set_state(missile_state)
        self.target.set_state(target_state)
        self.los.set_state(los_state)

        s = np.concatenate([los_state,
                            np.array([self.missile_etta, self.missile_beta], dtype=np.float32),
                            missile_state[2:],
                            target_state[2:],
                            np.array([missile_thrust], dtype=np.float32)])

        self.t += self.tau

        terminal, info = self._terminal()

        # reward = self.get_reward()
        reward = self.get_reward_new(los_ds[0]*self.tau, np.rad2deg(self.missile_etta), self.state[0])
        # reward = self.get_reward_new_new(distance, next_distance, info)

        appendix = np.concatenate([
            np.array([self.t, self.missile_etta, self.target_etta, self.missile_beta, reward], dtype=np.float32),
            los_state,
            missile_state,
            target_state
        ])
        self.log = np.append(self.log, [appendix], axis=0)

        self.state = s

        return (s, reward, terminal, info)

    def get_weight_lines(self, n):
        """
        for plot only
        """
        x = np.linspace(0, self.DETECTION_DISTANCE_MAX, 1000)
        line1 = np.array([], dtype=np.float32)
        for i in range(len(x)):
            line1 = np.append(line1, 1 - weights_foo(x[i], self.DETECTION_DISTANCE_MAX, n))
        line2 = 1 - line1
        return x, line1, line2

    def get_etta_reward_line(self, sigma, mu):
        x = np.linspace(-self.MISSILE_ETTA_MAX, self.MISSILE_ETTA_MAX, 200)
        line = np.array([], dtype=np.float32)
        for i in range(len(x)):
            line = np.append(line, 10 * normal_distribution(sigma, mu, x[i]))
        return x, line

    def get_distance_reward_line(self, vel, dt):
        high = self.observation_space.high[4] * dt
        x = np.linspace(-high, high, 200)
        line = np.array([], dtype=np.float32)
        for i in range(len(x)):
            line = np.append(line, parametric_sigmoid(vel*self.K, dt, x[i]))
        return x, line

    def get_reward_new(self, dr, etta, distance):
        missile_vel = self.state[4]
        sigma = 4
        etta_reference = 0
        mu = etta_reference
        etta_reward = 10 * normal_distribution(sigma, mu, etta)
        distance_reward = parametric_sigmoid(missile_vel*self.K, self.tau, dr)
        n = 0.77
        gamma1 = 1 - weights_foo(distance, self.DETECTION_DISTANCE_MAX, n)
        gamma2 = 1 - gamma1

        summary = etta_reward ** gamma1 * distance_reward ** gamma2

        self.reward_data = np.array(
            [summary, etta_reward, sigma, mu, distance_reward, dr, gamma1, n], dtype=np.float32
        )
        return summary

    @staticmethod
    def get_reward_new_new(distance, next_distance, info):
        if info == 'hit':
            return 10
        if info == 'out of range' or info == 'too slow' or info == 'too much':
            return -20
        return 1 - next_distance / distance

    def get_reward(self):
        gamma1 = 0.55
        gamma2 = 1 - gamma1
        u1 = 0.4
        u2 = 1 - u1

        if abs(self.missile_etta) > self.MISSILE_ETTA_MAX/2:
            missile_etta_factor = 0
        else:
            missile_etta_factor = np.exp(-abs(self.missile_etta) / self.MISSILE_ETTA_MAX)

        # if pi / 2 <= abs(self.target_etta) <= pi:
        #     target_etta_factor = np.cos(pi - abs(self.target_etta)) * np.exp(pi - abs(self.target_etta))
        # else:
        #     target_etta_factor = 0.5 * np.cos(abs(self.target_etta)) * np.exp(abs(self.target_etta))

        target_etta_factor = max(1 - abs(self.state[2] - self.opts['los']['chi0'])**0.5, 0)

        if self.state[0] > self.DETECTION_DISTANCE_MAX:
            distance_factor = 0
        else:
            distance_factor = np.exp(-100 * abs(self.state[0]) / self.DETECTION_DISTANCE_MAX)

        total_angle_factor = missile_etta_factor ** gamma1 * target_etta_factor ** gamma2
        total_detection_factor = total_angle_factor ** u1 * distance_factor ** u2

        return total_detection_factor*10

    def _terminal(self):
        assert self.state is not None, "Call reset before using AirCombat2DEnv object."

        if self.state[0] < self.EXPLOSION_DISTANCE:
            info = 'hit'
            done = True
        elif self.state[2] > np.radians(self.MISSILE_ETTA_MAX) or self.state[2] < -np.radians(self.MISSILE_ETTA_MAX):
            info = 'out of range'
            done = True
        elif self.state[4] < self.missile.MIN_VELOCITY:
            info = 'too slow'
            done = True
        elif self.t > self.TERMINATION_TIME:
            info = 'too much'
            done = True
        else:
            info = None
            done = False
        return done, info

    def render(self, mode="human"):
        fig = figure(figsize=(10, 9))
        plt.subplot2grid((8, 9), (0, 0), colspan=5, rowspan=5)
        los_p1 = [self.log[:, 7][-1], self.log[:, 11][-1]]
        los_p2 = [self.log[:, 8][-1], self.log[:, 12][-1]]
        plt.plot(los_p1, los_p2, c='black', ls='--', label='los')
        plt.plot(self.log[:, 7], self.log[:, 8], c='red', label='missile', lw=4)
        plt.plot(self.log[:, 11], self.log[:, 12], c='blue', label='target', lw=4)
        plt.xlabel('$x$, м')
        plt.ylabel('$z$, м')
        plt.axis('equal')
        l1 = plt.legend(
            [f"t = {round(self.log[:, 0][-1], 2)} c\n"
             f"r = {round(self.log[:, 5][-1], 0)} м\n"
             f"reward = {round(self.log[:, 4][-1], 4)}"
             ],
            loc='upper left',
            scatterpoints=1,
            handlelength=0,
            handletextpad=0
        )
        l1.get_frame().set_facecolor('none')
        l1.get_frame().set_linewidth(0.0)
        plt.legend(loc='upper right')
        plt.gca().add_artist(l1)
        plt.grid()

        plt.subplot2grid((8, 9), (0, 6), colspan=3, rowspan=2)
        plt.plot(self.log[:, 0], self.log[:, 9], c='tomato', label='missile velocity')
        plt.plot(self.log[:, 0], self.log[:, 13], c='dodgerblue', label='target velocity')
        plt.xlabel('$t$, с')
        plt.ylabel('$v$, м/с')
        plt.legend(loc='upper left')
        plt.grid()

        plt.subplot2grid((8, 9), (3, 6), colspan=3, rowspan=2)
        plt.plot(self.log[:, 0], np.rad2deg(self.log[:, 3]), c='mediumseagreen', label='missile β')
        plt.plot(self.log[:, 0], np.rad2deg(self.log[:, 2]), c='sandybrown', label='target η')
        plt.plot(self.log[:, 0], np.rad2deg(self.log[:, 1]), c='mediumpurple', label='missile η')
        plt.xlabel('$t$, с')
        plt.ylabel('$∠$, °')
        plt.legend(loc='upper left')
        plt.grid()

        summary, etta_reward, sigma, mu, distance_reward, dr, gamma1, n = self.reward_data

        plt.subplot2grid((8, 9), (6, 0), colspan=3, rowspan=2)
        x, line1, line2 = self.get_weight_lines(n)
        plt.plot(x, line1, label=r'$\gamma$$_1$')
        plt.plot(x, line2, label=r'$\gamma$$_2$')
        plt.scatter(self.log[:, 5][-1], gamma1)
        plt.scatter(self.log[:, 5][-1], 1 - gamma1)
        plt.xlabel('$r$, м')
        plt.legend(loc='center right')
        plt.grid()

        plt.subplot2grid((8, 9), (6, 4), colspan=2, rowspan=2)
        x, line = self.get_etta_reward_line(sigma, mu)
        plt.plot(x, line)
        plt.scatter(np.rad2deg(self.log[:, 1][-1]), etta_reward)
        plt.xlabel('$η$, град')
        plt.ylabel('$η$$_{rew}$')
        plt.grid()

        plt.subplot2grid((8, 9), (6, 7), colspan=2, rowspan=2)
        x, line = self.get_distance_reward_line(self.log[:, 9][-1], self.tau)
        plt.plot(x, line)
        plt.scatter(dr, distance_reward)
        plt.xlabel('$dr$, м')
        plt.ylabel('$dr_{rew}$')
        plt.grid()

        plt.show()
        clear_output(True)
