# Здесь будут лежать методы наведения


def pn(k, delta_phi, delta_chi):
    """
    Proportional Navigation (метод пропорционального сближения)

    :param k: коэффициент пропорциональности
    :param delta_phi: dφ/dt
    :param delta_chi: dχ/dt
    :return: dΘ/dt, dΨ/dt
    """
    return k * delta_phi, k * delta_chi
