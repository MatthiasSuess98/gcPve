import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    info = np.loadtxt("raw/GPU_Info.csv", delimiter=";", dtype="str")
    prop = np.loadtxt("raw/Bench_Prop.csv", delimiter=";", dtype="str")
    derivatives = np.loadtxt("raw/InfoProp_Derivatives.csv", delimiter=";", dtype="str")
    warpSize = int(info[5, 1].replace('"', ''))
    multiProcessorCount = int(info[16, 1].replace('"', ''))
    hardwareWarpsPerSm = int(derivatives[3, 1].replace('"', ''))

    data = np.loadtxt("raw/Benchmark_L1.csv", delimiter=";", dtype="double")
    lane = range(warpSize)
    for x in range(multiProcessorCount):
        fig = plt.figure()
        for y in range(hardwareWarpsPerSm):
            if (y % 4) == 1:
                graphColor = 'red'
            elif (y % 4) == 2:
                graphColor = 'green'
            elif (y % 4) == 3:
                graphColor = 'blue'
            else:
                graphColor = 'orange'
            time = data[((x * hardwareWarpsPerSm) + y), :]
            plt.stem(lane, time, markerfmt=graphColor, label=('Hardware warp ' + str(y)))
        plt.title('Benchmark: Average load time of data in the L1 cache of SM ' + str(x) + '.')
        plt.xlabel("Lane id")
        plt.ylabel("Load time in ns")
        plt.legend()
        fig.savefig('out/Benchmark_L1_SM' + str(x) + '.jpeg', bbox_inches='tight')

    data = np.loadtxt("raw/Benchmark_SM.csv", delimiter=";", dtype="double")
    lane = range(warpSize)
    for x in range(multiProcessorCount):
        fig = plt.figure()
        for y in range(hardwareWarpsPerSm):
            if (y % 4) == 1:
                graphColor = 'red'
            elif (y % 4) == 2:
                graphColor = 'green'
            elif (y % 4) == 3:
                graphColor = 'blue'
            else:
                graphColor = 'orange'
            time = data[((x * hardwareWarpsPerSm) + y), :]
            plt.stem(lane, time, markerfmt=graphColor, label=('Hardware warp ' + str(y)))
        plt.title('Benchmark: Average load time of data in the shared memory of SM ' + str(x) + '.')
        plt.xlabel("Lane id")
        plt.ylabel("Load time in ns")
        plt.legend()
        fig.savefig('out/Benchmark_SM_SM' + str(x) + '.jpeg', bbox_inches='tight')

    data = np.loadtxt("raw/Benchmark_L2.csv", delimiter=";", dtype="double")
    lane = range(warpSize)
    for x in range(multiProcessorCount):
        fig = plt.figure()
        for y in range(hardwareWarpsPerSm):
            if (y % 4) == 1:
                graphColor = 'red'
            elif (y % 4) == 2:
                graphColor = 'green'
            elif (y % 4) == 3:
                graphColor = 'blue'
            else:
                graphColor = 'orange'
            time = data[((x * hardwareWarpsPerSm) + y), :]
            plt.stem(lane, time, markerfmt=graphColor, label=('Hardware warp ' + str(y)))
        plt.title('Benchmark: Average load time of data in the L2 cache of SM ' + str(x) + '.')
        plt.xlabel("Lane id")
        plt.ylabel("Load time in ns")
        plt.legend()
        fig.savefig('out/Benchmark_L2_SM' + str(x) + '.jpeg', bbox_inches='tight')

    data = np.loadtxt("raw/Benchmark_GM.csv", delimiter=";", dtype="double")
    lane = range(warpSize)
    for x in range(multiProcessorCount):
        fig = plt.figure()
        for y in range(hardwareWarpsPerSm):
            if (y % 4) == 1:
                graphColor = 'red'
            elif (y % 4) == 2:
                graphColor = 'green'
            elif (y % 4) == 3:
                graphColor = 'blue'
            else:
                graphColor = 'orange'
            time = data[((x * hardwareWarpsPerSm) + y), :]
            plt.stem(lane, time, markerfmt=graphColor, label=('Hardware warp ' + str(y)))
        plt.title('Benchmark: Average load time of data in the global memory of SM ' + str(x) + '.')
        plt.xlabel("Lane id")
        plt.ylabel("Load time in ns")
        plt.legend()
        fig.savefig('out/Benchmark_GM_SM' + str(x) + '.jpeg', bbox_inches='tight')

    quit()


# FINISHED

