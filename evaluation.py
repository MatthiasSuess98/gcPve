import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt("Benchmark_1.csv", delimiter=";", dtype="int")
    smId = data[:, 0]
    time = data[:, 1]
    fig = plt.figure()
    plt.plot(smId, time, color='green')
    plt.title("Benchmark 1:")
    plt.xlabel("streaming multiprocessor")
    plt.ylabel("computation time")
    fig.savefig('Benchmark_1.pdf', bbox_inches='tight')
    plt.show()

    ## 1. benchmark: all 30 sm
    ## 2. benchmark: computation time per array size for one sm
    ## 3. benchmark: differences in one warp of one sm

