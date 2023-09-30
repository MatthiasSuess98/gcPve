#ifndef GCPVE_10_PERFORM_BENCHMARK_CUH
#define GCPVE_10_PERFORM_BENCHMARK_CUH

#include <vector>

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"

#include "11-Sort_Data_Collection.cuh"

#include "20-Launch_Benchmark.cuh"


/**
 * Performs the benchmark.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 * @return The final core characteristics from the benchmark.
 */
std::vector<CoreCharacteristics> performBenchmarks(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    // Initialize data collection.
    dataCollection data;
    for (int i = 0; i < (info.multiProcessorCount * derivatives.hardwareWarpsPerSm * prop.collectionFactor * info.warpSize); i++) {
        int mulpIni = 0;
        data.mulp.push_back(mulpIni);
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
    CoreCharacteristics gpuCore = CoreCharacteristics(0, 0, 0);
    for (int i = 0; i < info.multiProcessorCount; i++) {
        for (int j = 0; j < derivatives.hardwareWarpsPerSm; j++) {
            for (int k = 0; k < info.warpSize; k++) {
                gpuCore = CoreCharacteristics(i, j, k);
                gpuCores.push_back(gpuCore);
            }
        }
    }

    // Perform the benchmark loop.
    for (int i = 0; i < prop.numberOfTrialsPerform; i++) {

        // Reset the data collection.
        for (int j = 0; j < (info.multiProcessorCount * derivatives.hardwareWarpsPerSm * prop.collectionFactor * info.warpSize); j++) {
            data.mulp[j] = 0;
            data.lane[j] = 0;
            data.timeL1[j] = 0.0;
            data.timeSM[j] = 0.0;
            data.timeL2[j] = 0.0;
            data.timeGM[j] = 0.0;
        }

        // Launches all four benchmarks.
        data = launchBenchmarks(info, prop, derivatives, data);

        // Sort the resulted data into the core characteristics.
        gpuCores = sortDataCollection(info, prop, derivatives, data, gpuCores);
    }

    return gpuCores;
}


/**
 * Print the results of the benchmarks into separate csv files.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 * @param benchmark The core characteristics from the benchmarks.
 */
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

