#ifndef GCPVE_20_L1_CACHE_LAUNCHER_CUH
#define GCPVE_20_L1_CACHE_LAUNCHER_CUH

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"
#include "21-L1_Cache_Benchmark.cuh"

/**
 * Function that performs a L1 benchmark for data collections.
 * It uses a shotgun technique to get the full data.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 * @return A complete fully sorted data collection for the L1 cache.
 */
DataCollection performL1Benchmark(GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    // Collection A: For host use only.
    DataCollection finalCollection;
    for (int i = 0; i < derivatives.collectionSize; i++) {
        int iniMulp = 0;
        finalCollection.mulp.push_back(iniMulp);
        int iniWarp = 0;
        finalCollection.warp.push_back(iniWarp);
        int iniLane = 0;
        finalCollection.lane.push_back(iniLane);
        long long int iniTime = 0;
        finalCollection.time.push_back(iniTime);
    }

    // Collection B: For both host use and device use.
    DataCollection benchCollection;
    for (int i = 0; i < derivatives.collectionSize; i++) {
        int iniMulp = 0;
        benchCollection.mulp.push_back(iniMulp);
        int iniWarp = 0;
        benchCollection.warp.push_back(iniWarp);
        int iniLane = 0;
        benchCollection.lane.push_back(iniLane);
        long long int iniTime = 0;
        benchCollection.time.push_back(iniTime);
    }

    // Allocation of collection B to the global device memory.
    dataCollection *ptr;
    ptr = &benchCollection;
    cudaMallocManaged(&ptr, sizeof(benchCollection));

    // Getting and sorting the data.
    bool moveOn;
    for (int mulpLoop = 0; mulpLoop < info.multiProcessorCount; mulpLoop++) {
        moveOn = true;
        for (int trailLoop = 0; moveOn && (trailLoop < prop.numberOfTrialsLaunch); trailLoop++) {
            for (int resetLoop = 0; resetLoop < derivatives.collectionSize; resetLoop++) {
                (*ptr).mulp[resetLoop] = 0;
                (*ptr).warp[resetLoop] = 0;
                (*ptr).lane[resetLoop] = 0;
                (*ptr).time[resetLoop] = 0;
            }
            launchL1Benchmarks(ptr, info, prop, derivatives);
            for (int blockLoop = 0; moveOn && (blockLoop < derivatives.NumberOfBlocks); blockLoop++) {
                if (moveOn && ((*ptr).mulp[blockLoop * info.warpSize] == mulpLoop) && ((*ptr).time[blockLoop * info.warpSize] != 0)) {
                    for (int freeLoop = 0; moveOn && (freeLoop < derivatives.NumberOfBlocksPerMulp); freeLoop++) {
                        if (moveOn && (finalCollection.time[((mulpLoop * derivatives.NumberOfBlocksPerMulp) + freeLoop) * info.warpSize] == 0)) {
                            for (int laneLoop = 0; moveOn && (laneLoop < info.warpSize); laneLoop++) {
                                finalCollection.mulp[(((mulpLoop * derivatives.NumberOfBlocksPerMulp) + freeLoop) * info.warpSize) + laneLoop] = (*ptr).mulp[(blockLoop * info.warpSize) + laneLoop];
                                finalCollection.warp[(((mulpLoop * derivatives.NumberOfBlocksPerMulp) + freeLoop) * info.warpSize) + laneLoop] = (*ptr).warp[(blockLoop * info.warpSize) + laneLoop];
                                finalCollection.lane[(((mulpLoop * derivatives.NumberOfBlocksPerMulp) + freeLoop) * info.warpSize) + laneLoop] = (*ptr).lane[(blockLoop * info.warpSize) + laneLoop];
                                finalCollection.time[(((mulpLoop * derivatives.NumberOfBlocksPerMulp) + freeLoop) * info.warpSize) + laneLoop] = (*ptr).time[(blockLoop * info.warpSize) + laneLoop];
                            }
                        }
                        if (moveOn && (finalCollection.time[((mulpLoop + 1) * derivatives.NumberOfBlocksPerMulp * info.warpSize) - 1] != 0)) {
                            moveOn = false;
                        }
                    }
                }
            }
        }
        // If the maximum of trails is reached and some sets could not be filled: Print error.
        if (moveOn) {
            printf("[ERROR] Failed to get full l1 data for streaming multiprocessor %d in benchmark.\n", mulpLoop);
        }
    }

    // Free the allocated global device memory of collection B.
    cudaFree(ptr);

    //Return the data collection A with the final benchmark data.
    return finalCollection;
}

#endif //GCPVE_20_L1_CACHE_LAUNCHER_CUH

//FINISHED

