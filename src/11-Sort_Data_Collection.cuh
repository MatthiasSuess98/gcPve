#ifndef GCPVE_11_SORT_DATA_COLLECTION_CUH
#define GCPVE_11_SORT_DATA_COLLECTION_CUH

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
std::vector<CoreCharacteristics> sortDataCollection(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives, dataCollection data, std::vector<CoreCharacteristics> gpuCores) {


    for (int blockLoop = 0; blockLoop < data.collectionSize; blockLoop = blockLoop + info.warpSize) {
        if (data.timeL1[blockLoop] != 0) {
            hardwareWarpScore = 0;
            for (int hardwareWarpLoop = 0; hardwareWarpLoop < derivatives.hardwareWarpsPerSm; hardwareWarpLoop++) {
                dontFits[hardwareWarpLoop] = 0;
            }
            for (int hardwareWarpLoop = 0; hardwareWarpLoop < derivatives.hardwareWarpsPerSm; hardwareWarpLoop++) {
                if (gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + (hardwareWarpLoop * info.warpSize)].getTypicalL1Time() != 0.0) {
                    hardwareWarpScore++;
                }
            }
            if (hardwareWarpScore == 0) {
                for (int laneLoop = 0; laneLoop < info.warpSize; laneLoop++) {
                    gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + laneLoop].setTypicalL1Time(((long double) data.time[blockLoop + laneLoop]) / ((long double) (derivatives.smallNumberOfTrialsDivisor * derivatives.smallNumberOfTrialsDivisor)));
                }
            } else if (hardwareWarpScore == derivatives.hardwareWarpsPerSm) {
                for (int hardwareWarpLoop = 0; hardwareWarpLoop < derivatives.hardwareWarpsPerSm; hardwareWarpLoop++) {
                    for (int laneLoop = 0; laneLoop < info.warpSize; laneLoop++) {
                        if (std::abs(gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + (hardwareWarpLoop * info.warpSize) + laneLoop].getTypicalL1Time() - (((long double) data.time[blockLoop]) / ((long double) (derivatives.smallNumberOfTrialsDivisor * derivatives.smallNumberOfTrialsDivisor)))) >= prop.maxDelta) {
                            dontFits[hardwareWarpLoop]++;
                        }
                    }
                }
                smallestNumber = 0;
                bestHardwareWarp = 0;
                for (int hardwareWarpLoop = 0; hardwareWarpLoop < derivatives.hardwareWarpsPerSm; hardwareWarpLoop++) {
                    if (hardwareWarpLoop == 0) {
                        smallestNumber = dontFits[hardwareWarpLoop];
                    } else {
                        if (smallestNumber > dontFits[hardwareWarpLoop]) {
                            smallestNumber = dontFits[hardwareWarpLoop];
                        }
                    }
                }
                for (int hardwareWarpLoop = 0; hardwareWarpLoop < derivatives.hardwareWarpsPerSm; hardwareWarpLoop++) {
                    if (smallestNumber == dontFits[hardwareWarpLoop]) {
                        bestHardwareWarp = hardwareWarpLoop;
                    }
                }
                for (int laneLoop = 0; laneLoop < info.warpSize; laneLoop++) {
                    currentTime = gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + laneLoop].getTypicalL1Time();
                    gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + laneLoop].setTypicalL1Time(((((long double) data.time[blockLoop + laneLoop]) / ((long double) (derivatives.smallNumberOfTrialsDivisor * derivatives.smallNumberOfTrialsDivisor))) + currentTime) / 2);
                }
            } else {
                for (int hardwareWarpLoop = 0; hardwareWarpLoop < hardwareWarpScore; hardwareWarpLoop++) {
                    for (int laneLoop = 0; laneLoop < info.warpSize; laneLoop++) {
                        if (std::abs(gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + (hardwareWarpLoop * info.warpSize) + laneLoop].getTypicalL1Time() - (((long double) data.time[blockLoop]) / ((long double) (derivatives.smallNumberOfTrialsDivisor * derivatives.smallNumberOfTrialsDivisor)))) >= prop.maxDelta) {
                            dontFits[hardwareWarpLoop]++;
                        }
                    }
                }
                smallestNumber = 0;
                bestHardwareWarp = 0;
                for (int hardwareWarpLoop = 0; hardwareWarpLoop < hardwareWarpScore; hardwareWarpLoop++) {
                    if (hardwareWarpLoop == 0) {
                        smallestNumber = dontFits[hardwareWarpLoop];
                    } else {
                        if (smallestNumber > dontFits[hardwareWarpLoop]) {
                            smallestNumber = dontFits[hardwareWarpLoop];
                        }
                    }
                }
                for (int hardwareWarpLoop = 0; hardwareWarpLoop < hardwareWarpScore; hardwareWarpLoop++) {
                    if (smallestNumber == dontFits[hardwareWarpLoop]) {
                        bestHardwareWarp = hardwareWarpLoop;
                    }
                }
                if (dontFits[bestHardwareWarp] > prop.maxDontFit) {
                    for (int laneLoop = 0; laneLoop < info.warpSize; laneLoop++) {
                        gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + (hardwareWarpScore * info.warpSize) + laneLoop].setTypicalL1Time(((long double) data.time[blockLoop + laneLoop]) / ((long double) (derivatives.smallNumberOfTrialsDivisor * derivatives.smallNumberOfTrialsDivisor)));
                    }
                } else {
                    for (int laneLoop = 0; laneLoop < info.warpSize; laneLoop++) {
                        currentTime = gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + laneLoop].getTypicalL1Time();
                        gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + (bestHardwareWarp * info.warpSize) + laneLoop].setTypicalL1Time(((((long double) data.time[blockLoop + laneLoop]) / ((long double) (derivatives.smallNumberOfTrialsDivisor * derivatives.smallNumberOfTrialsDivisor))) + currentTime) / 2);
                    }
                }
            }
        }
    }
}

#endif //GCPVE_11_SORT_DATA_COLLECTION_CUH

