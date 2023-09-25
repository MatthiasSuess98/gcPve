import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt("raw/Benchmark_L1.csv", delimiter=";", dtype="double")
    print(data)
    #smId = data[:, 0]
    #time = data[:, 1]
    #fig = plt.figure()
    #plt.step(smId, time, color='green')
    #plt.title("Benchmark: Average computation time of a simple add statement on sm 0.")
    #plt.xlabel("thread")
    #plt.ylabel("computation time in ns")
    #fig.savefig('Benchmark_16bit.pdf', bbox_inches='tight')
    quit()

#FINISHED

