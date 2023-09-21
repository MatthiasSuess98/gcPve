#ifndef GCPVE_20_L1_CACHE_LAUNCHER_CUH
#define GCPVE_20_L1_CACHE_LAUNCHER_CUH

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"

#include "21-L1_Cache_Benchmark.cuh"

/**
 * Function that performs a L1 benchmark for small data collections.
 * It uses a shotgun technique to get the full data.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 * @return A complete fully sorted small data collection for the L1 cache.
 */
SmallDataCollection performSmallL1Benchmark(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    // Collection A: For host use only.
    SmallDataCollection finalCollection;

    // Collection B: For both host use and device use.
    SmallDataCollection benchCollection;

    // Initialize Collection A.
    for (int initializeLoop = 0; initializeLoop < prop.small; initializeLoop++) {
        finalCollection.mulp[initializeLoop] = 0;
        finalCollection.warp[initializeLoop] = 0;
        finalCollection.lane[initializeLoop] = 0;
        finalCollection.time[initializeLoop] = 0;
    }

    // Allocation of collection B to the global device memory.
    SmallDataCollection *ptr;
    ptr = &benchCollection;
    cudaMallocManaged(&ptr, sizeof(benchCollection));

    // Getting and sorting the data.
    bool moveOn;
    for (int mulpLoop = 0; mulpLoop < info.multiProcessorCount; mulpLoop++) {
        moveOn = true;
        for (int trailLoop = 0; moveOn && (trailLoop < prop.numberOfTrialsLaunch); trailLoop++) {
            for (int resetLoop = 0; resetLoop < prop.small; resetLoop++) {
                (*ptr).mulp[resetLoop] = 0;
                (*ptr).warp[resetLoop] = 0;
                (*ptr).lane[resetLoop] = 0;
                (*ptr).time[resetLoop] = 0;
            }
            launchSmallL1Benchmarks(ptr, info, prop, derivatives);
            for (int blockLoop = 0; moveOn && (blockLoop < derivatives.smallNumberOfBlocks); blockLoop++) {
                if (moveOn && ((*ptr).mulp[blockLoop * info.warpSize] == mulpLoop) && ((*ptr).time[blockLoop * info.warpSize] != 0)) {
                    for (int freeLoop = 0; moveOn && (freeLoop < derivatives.smallNumberOfBlocksPerMulp); freeLoop++) {
                        if (moveOn && (finalCollection.time[(mulpLoop * derivatives.smallNumberOfBlocksPerMulp) + freeLoop] != 0)) {
                            for (int laneLoop = 0; moveOn && (laneLoop < info.warpSize); laneLoop++) {
                                finalCollection.mulp[(((mulpLoop * derivatives.smallNumberOfBlocksPerMulp) + freeLoop) * info.warpSize) + laneLoop] = (*ptr).mulp[(blockLoop * info.warpSize) + laneLoop];
                                finalCollection.warp[(((mulpLoop * derivatives.smallNumberOfBlocksPerMulp) + freeLoop) * info.warpSize) + laneLoop] = (*ptr).warp[(blockLoop * info.warpSize) + laneLoop];
                                finalCollection.lane[(((mulpLoop * derivatives.smallNumberOfBlocksPerMulp) + freeLoop) * info.warpSize) + laneLoop] = (*ptr).lane[(blockLoop * info.warpSize) + laneLoop];
                                finalCollection.time[(((mulpLoop * derivatives.smallNumberOfBlocksPerMulp) + freeLoop) * info.warpSize) + laneLoop] = (*ptr).time[(blockLoop * info.warpSize) + laneLoop];
                            }
                        }
                        if (moveOn && (finalCollection.time[((mulpLoop + 1) * derivatives.smallNumberOfBlocksPerMulp) - 1] != 0)) {
                            moveOn = false;
                        }
                    }
                }
            }
        }
        // If the maximum of trails is reached and some sets could not be filled: Print error.
        if (moveOn && (finalCollection.time[((mulpLoop + 1) * derivatives.smallNumberOfBlocksPerMulp) - 1] == 0)) {
            printf("[ERROR] Failed to get full l1 data for streaming multiprocessor %d in small benchmark.", mulpLoop);
        }
    }

    // Free the allocated global device memory of collection B.
    cudaFree(ptr);

    //Return the data collection A with the final benchmark data.
    return finalCollection;
}

#endif //GCPVE_20_L1_CACHE_LAUNCHER_CUH

//FINISHED

