#ifndef GCPVE_10_PERFORM_BENCHMARK_CUH
#define GCPVE_10_PERFORM_BENCHMARK_CUH

#include <vector>

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"
#include "20-L1_Cache_Launcher.cuh"

/**
 * Performs the small benchmark.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 */
std::vector<CoreCharacteristics> performBenchmarks(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    // Initialize data collection.
    dataCollection data;
    data.collectionSize = info.multiProcessorCount * derivatives.hardwareWarpsPerSm * info.warpSize * prop.collectionFactor;
    for (int initialLoop = 0; initialLoop < data.collectionSize; initialLoop++) {
        int mulpIni = 0;
        data.mulp.push_back(mulpIni);
        int warpIni = 0;
        data.warp.push_back(warpIni);
        int laneIni = 0;
        data.lane.push_back(laneIni);
        float timeL1Ini = 0.0;
        data.timeL1.push_back(timeL1Ini);
        float timeSmIni = 0.0;
        data.timeSM.push_back(timeSmIni);
        float timeL2Ini = 0.0;
        data.timeL2.push_back(timeL2Ini);
        float timeGmIni = 0.0;
        data.timeGM.push_back(timeGmIni);
    }


    // Declare and initialize all core characteristics.
    std::vector<CoreCharacteristics> gpuCores;
    CoreCharacteristics gpuCore;
    for (int i = 0; i < info.multiProcessorCount; i++) {
        for (int j = 0; j < derivatives.hardwareWarpsPerSm; j++) {
            for (int k = 0; k < info.warpSize; k++) {
                gpuCore = CoreCharacteristics(info.multiProcessorCount, derivatives.hardwareWarpsPerSm, info.warpSize, i, j, k);
                gpuCores.push_back(gpuCore);
            }
        }
    }

    // Perform the benchmark loop.
    for (int trailLoop = 0; trailLoop < prop.numberOfTrialsPerform; trailLoop++) {

        for (int resetLoop = 0; resetLoop < data.collectionSize; resetLoop++) {
            data.mulp[resetLoop] = 0;
            data.warp[resetLoop] = 0;
            data.lane[resetLoop] = 0;
            data.timeL1[resetLoop] = 0.0;
            data.timeSM[resetLoop] = 0.0;
            data.timeL2[resetLoop] = 0.0;
            data.timeGM[resetLoop] = 0.0;
        }

        data = launchL1Benchmark(info, prop, derivatives, data);
        data = launchSMBenchmark(info, prop, derivatives, data);
        data = launchL2Benchmark(info, prop, derivatives, data);
        data = launchGMBenchmark(info, prop, derivatives, data);

        gpuCores = sortDataCollection(info, prop, derivatives, data, gpuCores);
    }

    return gpuCores;
}


void createBenchmarkFile(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives, std::vector<CoreCharacteristics> benchmark) {

    // Create files with all benchmark data.
    char outputL1[] = "raw/Benchmark_L1.csv";
    char outputSm[] = "raw/Benchmark_SM.csv";
    char outputL2[] = "raw/Benchmark_L2.csv";
    char outputGm[] = "raw/Benchmark_GM.csv";
    FILE *csvL1 = fopen(outputL1, "w");
    FILE *csvSm = fopen(outputSm, "w");
    FILE *csvL2 = fopen(outputL2, "w");
    FILE *csvGm = fopen(outputGm, "w");
    for (int i = 0; i < info.multiProcessorCount; i++) {
        for (int j = 0; j < derivatives.hardwareWarpsPerSm; j++) {
            for (int k = 0; k < info.warpSize; k++) {
                fprintf(csvL1, "%f", gpuCores[(i * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize) + k].getTypicalL1Time());
                fprintf(csvSm, "%f", gpuCores[(i * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize) + k].getTypicalSmTime());
                fprintf(csvL2, "%f", gpuCores[(i * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize) + k].getTypicalL2Time());
                fprintf(csvGm, "%f", gpuCores[(i * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize) + k].getTypicalGmTime());
                if (k < (info.warpSize - 1)) {
                    fprintf(csvL1, ";");
                    fprintf(csvSm, ";");
                    fprintf(csvL2, ";");
                    fprintf(csvGm, ";");
                }
            }
            fprintf(csvL1, "\n");
            fprintf(csvSm, "\n");
            fprintf(csvL2, "\n");
            fprintf(csvGm, "\n");
        }
        fprintf(csvL1, "\n");
        fprintf(csvSm, "\n");
        fprintf(csvL2, "\n");
        fprintf(csvGm, "\n");
    }
    fclose(csvL1);
    fclose(csvSm);
    fclose(csvL2);
    fclose(csvGm);
    printf("[INFO] The benchmark files were created.\n");
}

#endif //GCPVE_10_PERFORM_BENCHMARK_CUH

//FINISHED

