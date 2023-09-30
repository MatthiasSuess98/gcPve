#ifndef GCPVE_11_SORT_DATA_COLLECTION_CUH
#define GCPVE_11_SORT_DATA_COLLECTION_CUH

#include <vector>

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"


/**
 * Sorts the data from the benchmark into the core characteristics.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 * @param data The data from the benchmark.
 * @param gpuCores The core characteristics in which the data from the benchmarks will be sorted into.
 * @return The core characteristics after the sorting process.
 */
std::vector<CoreCharacteristics> sortDataCollection(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives, dataCollection data, std::vector<CoreCharacteristics> gpuCores) {

    // Variables for the algorithm.
    bool rewrite = false;
    float currentTime = 0.0;
    int hardwareWarpScore = 0;
    int smallestNumber = 0;
    int bestHardwareWarp = 0;

    // Declaration and initialization of the dontFits counter.
    int dontFit;
    std::vector<int> dontFits;
    for (int i = 0; i < derivatives.hardwareWarpsPerSm; i++) {
        dontFit = 0;
        dontFits.push_back(dontFit);
    }

    // Sorting loop.
    for (int i = 0; i < (info.multiProcessorCount * derivatives.hardwareWarpsPerSm * prop.collectionFactor * info.warpSize); i = i + info.warpSize) {

        // Check whether the data it relevant.
        if ((data.timeL1[i] != 0) || (data.timeSM[i] != 0) || (data.timeL2[i] != 0) || (data.timeGM[i] != 0)) {

            // Reset all variables and the dontFits counter.
            rewrite = true;
            currentTime = 0.0;
            hardwareWarpScore = 0;
            smallestNumber = 0;
            bestHardwareWarp = 0;
            for (int j = 0; j < derivatives.hardwareWarpsPerSm; j++) {
                dontFits[j] = 0;
            }

            // Calculate the hardware score which is the number of hardware warps which are already initialized.
            for (int j = 0; j < derivatives.hardwareWarpsPerSm; j++) {
                if ((gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize)].getTypicalL1Time() != 0.0) || (gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize)].getTypicalSmTime() != 0.0) || (gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize)].getTypicalL2Time() != 0.0) || (gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize)].getTypicalGmTime() != 0.0)) {
                    hardwareWarpScore++;
                }
            }

            // Calculate the dontFits of all hardware warps of all memory types.
            for (int j = 0; j < hardwareWarpScore; j++) {
                for (int k = 0; k < info.warpSize; k++) {
                    if (std::abs(gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize) + k].getTypicalL1Time() - data.timeL1[i + k]) >= prop.maxDelta) {
                        dontFits[j]++;
                    }
                    if (std::abs(gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize) + k].getTypicalSmTime() - data.timeSM[i + k]) >= prop.maxDelta) {
                        dontFits[j]++;
                    }
                    if (std::abs(gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize) + k].getTypicalL2Time() - data.timeL2[i + k]) >= prop.maxDelta) {
                        dontFits[j]++;
                    }
                    if (std::abs(gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (j * info.warpSize) + k].getTypicalGmTime() - data.timeGM[i + k]) >= prop.maxDelta) {
                        dontFits[j]++;
                    }
                }
            }

            // Find the smallest number of all dontFits.
            for (int j = 0; j < hardwareWarpScore; j++) {
                if (j == 0) {
                    smallestNumber = dontFits[j];
                } else {
                    if (smallestNumber > dontFits[j]) {
                        smallestNumber = dontFits[j];
                    }
                }
            }

            // Find the best hardware warp.
            for (int j = 0; j < hardwareWarpScore; j++) {
                if (smallestNumber == dontFits[j]) {
                    bestHardwareWarp = j;
                }
            }

            // Decide how the data should be sorted in.
            if (hardwareWarpScore == 0) {
                rewrite = true;
            } else if (hardwareWarpScore == derivatives.hardwareWarpsPerSm) {
                rewrite = false;
            } else {
                if (dontFits[bestHardwareWarp] > prop.maxDontFit) {
                    rewrite = true;
                } else {
                    rewrite = false;
                }
            }

            // Sort the benchmark data into the core characteristics.
            if (rewrite) {
                for (int j = 0; j < info.warpSize; j++) {
                    gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (hardwareWarpScore * info.warpSize) + j].setTypicalL1Time(data.timeL1[i + j]);
                    gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (hardwareWarpScore * info.warpSize) + j].setTypicalSmTime(data.timeSM[i + j]);
                    gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (hardwareWarpScore * info.warpSize) + j].setTypicalL2Time(data.timeL2[i + j]);
                    gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (hardwareWarpScore * info.warpSize) + j].setTypicalGmTime(data.timeGM[i + j]);
                }
            } else {
                printf("%d", data.mulp[i]);
                for (int j = 0; j < info.warpSize; j++) {
                    currentTime = gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + j].getTypicalL1Time();
                    gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + j].setTypicalL1Time((data.timeL1[i + j] + currentTime) / 2);
                    currentTime = gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + j].getTypicalSmTime();
                    gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + j].setTypicalSmTime((data.timeSM[i + j] + currentTime) / 2);
                    currentTime = gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + j].getTypicalL2Time();
                    gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + j].setTypicalL2Time((data.timeL2[i + j] + currentTime) / 2);
                    currentTime = gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + j].getTypicalGmTime();
                    gpuCores[(data.mulp[i] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + j].setTypicalGmTime((data.timeGM[i + j] + currentTime) / 2);
                }
            }
        }
    }

    // Return the new core characteristics.
    return gpuCores;
}


#endif //GCPVE_11_SORT_DATA_COLLECTION_CUH

//FINISHED

292929292929292929292929292929292929292929292929292929292929292929292929292929
29292929292929292929292929292929292929292929292929292929292929292929292929292929