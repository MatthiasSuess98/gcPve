import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt("Benchmark.csv", delimiter=";", dtype="int")
    threadId = data[:, 0]
    blockId = data[:, 1]
    laneId = data[:, 2]
    warpId = data[:, 3]
    smId = data[:, 4]
    begin = data[:, 5]
    end = data[:, 6]
    core = laneId + (warpId * 32) + (smId * 128)
    timeDifference = end - begin
    fig = plt.figure()
    test = np.arange(0, 65536)
    plt.plot(test, timeDifference, color='red')
    plt.title("Intensities after direct light")
    plt.xlabel("Wavelength in nm")
    plt.ylabel("Intensity in counts")
    fig.savefig('Intensities2.pdf', bbox_inches='tight')
    plt.show()



    ##np.savetxt("Benchmark_new.csv", data, delimiter=" ; ", newline=" \n", fmt="%d")

    print("threadId ; blockId ; laneId ; warpId ; smId ; begin ; end ; TimeDifference")
    print(core)
    print(timeDifference)
