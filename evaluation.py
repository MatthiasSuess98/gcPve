import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt("Benchmark_1.csv", delimiter=";", dtype="float")
    smId = data[:, 0]
    time = data[:, 1]
    fig = plt.figure()
    plt.plot(smId, time, color='green')
    plt.title("Benchmark 1: Computation time of all streaming multiprocessors (sm\'s)")
    plt.xlabel("streaming multiprocessor (sm)")
    plt.ylabel("computation time in ns")
    fig.savefig('Benchmark_1.pdf', bbox_inches='tight')
    plt.show()

    #data = np.loadtxt("Benchmark_2.csv", delimiter=";", dtype="float")
    #smId = data[:, 0]
    #time = data[:, 1]
    #fig = plt.figure()
    #plt.plot(smId, time, color='green')
    #plt.title("Benchmark 2: Computation time per array size for sm 0")
    #plt.xlabel("array size")
    #plt.ylabel("computation time in ns")
    #fig.savefig('Benchmark_2.pdf', bbox_inches='tight')
    #plt.show()

    data = np.loadtxt("Benchmark_3.csv", delimiter=";", dtype="float")
    smId = data[:, 0]
    time = data[:, 1]
    fig = plt.figure()
    plt.plot(smId, time, color='green')
    plt.title("Benchmark 3: Differences of computation time in warp 0 for sm 0")
    plt.xlabel("lane id inside warp 0")
    plt.ylabel("computation time in ns")
    fig.savefig('Benchmark_3.pdf', bbox_inches='tight')
    plt.show()

    ## 1. benchmark: all 30 sm
    ## 2. benchmark: computation time per array size for one sm
    ## 3. benchmark: differences in one warp of one sm

