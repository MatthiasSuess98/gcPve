import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.loadtxt("Benchmark_L1.csv", delimiter=";", dtype="double")
    lane = range(32)
    for x in range(30):
        fig = plt.figure()
        for y in range(4):
            if (y % 4) == 1:
                graphColor = 'red'
            elif (y % 4) == 2:
                graphColor = 'green'
            elif (y % 4) == 3:
                graphColor = 'blue'
            else:
                graphColor = 'orange'
            time = data[((x * 4) + y), :]
            plt.stem(lane, time, markerfmt=graphColor, label=('Hardware warp ' + str(y)))
        plt.title('Benchmark: Average load time of data in the L1 cache of SM ' + str(x) + '.')
        plt.xlabel("Lane ids")
        plt.ylabel("Computation time in ns")
        plt.legend()
        fig.savefig('out/Benchmark_L1_SM' + str(x) + '.jpeg', bbox_inches='tight')
    quit()

#FINISHED

