import numpy as np


def quadrature_test(weight):
    print(np.sum(weight))
    assert (np.abs(np.sum(weight) - 1) < 1E-5)


def fetch_SN_quadrature(quadrature_N):
    # weight_index are top to bottom, left to right
    if quadrature_N == 4:
        node = np.array([0.3500212, 0.8688903])
        weight_multiplier = np.array([1/3])
        weight_index = np.ones(3)
    elif quadrature_N == 6:
        node = np.array([0.2666355, 0.6815076, 0.9261808])
        weight_multiplier = np.array([0.1761263, 0.1572071])
        weight_index = np.array([1, 2, 2, 1, 2, 1])
    elif quadrature_N == 8:
        node = np.array([0.2182179, 0.5773503, 0.7867958, 0.9511897])
        weight_multiplier = np.array([0.1209877, 0.0907407, 0.0925926])
        weight_index = np.array([1, 2, 2, 2, 3, 2, 1, 2, 2, 1])
    elif quadrature_N == 12:
        node = np.array([0.1672126, 0.4595476, 0.6280191, 0.7600210, 0.8722706, 0.9716377])
        weight_multiplier = np.array([0.0707626, 0.0558811, 0.0373377, 0.0502819, 0.0258513])
        weight_index = np.array([1, 2, 2, 3, 4, 3, 3, 5, 5, 3, 2, 4, 5, 4, 2, 1, 2, 3, 3, 2, 1])
    elif quadrature_N == 16:
        node = np.array([0.1389568, 0.3922893, 0.5370966, 0.6504264, 0.7467506, 0.8319966, 0.9092855, 0.9805009])
        weight_multiplier = np.array([0.0489872, 0.0413296, 0.0212326, 0.0256207, 0.0360486, 0.0144589, 0.0344958, 0.0085179])
        weight_index = np.array([1, 2, 2, 3, 5, 3, 4, 6, 6, 4, 4, 7, 8, 7, 4, 3, 6, 8, 8, 6, 3, 2, 5, 6, 7, 6, 5, 2, 1, 2, 3, 4, 4, 3, 2, 1])
    else:
        raise Exception('Available quadratures: S4, S6, S8, S12, S16.')

    weight_index = (weight_index - 1).astype(int)
    points = int(quadrature_N*(quadrature_N+2)/8)
    mu = np.zeros(points)
    eta = np.zeros(points)
    weight = np.zeros(points)

    num_rows = int(quadrature_N / 2)
    mu_counter = 0
    eta_counter = 0

    for i in range(num_rows):
        for j in reversed(range(i+1)):
            mu[mu_counter] = node[j]
            mu_counter += 1
        for j in range(i+1):
            eta[eta_counter] = node[j]
            corresponding_weight_index = weight_index[eta_counter]
            weight[eta_counter] = weight_multiplier[corresponding_weight_index]
            eta_counter += 1
    return mu, eta, weight


mu, eta, weight = fetch_SN_quadrature(16)
quadrature_test(weight)
    # positive positive
    # negative positive
    # positive negative
    # negative negative