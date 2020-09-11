import numpy as np

if __name__ == "__main__":
    print()
    simulations = np.zeros(shape=(1000, 1000))

    result_1 = np.ones(shape=(1000,))

    simulations[0] = result_1
    simulations[2,::] = result_1
    simulations[4] = np.asarray(result_1)
    print()