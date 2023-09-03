import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt("../out/Benchmark_1.csv", delimiter=";", dtype="float")
    smId = data[:, 0]
    time = data[:, 1]
    fig = plt.figure()
    plt.plot(smId, time, color='green')
    plt.title("Benchmark 1: Average computation time of a 16777216-array")
    plt.xlabel("streaming multiprocessor (sm)")
    plt.ylabel("computation time in ns")
    fig.savefig('Benchmark_1.pdf', bbox_inches='tight')
    plt.show()

