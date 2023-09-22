#ifndef GCPVE_21_L1_CACHE_BENCHMARK_CUH
#define GCPVE_21_L1_CACHE_BENCHMARK_CUH

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"

/**
 * The L1 benchmark which uses the small data collection.
 * @param ptr The small data collection where the data of the benchmark is stored.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 */
__global__ void smallL1Benchmark(SmallDataCollection *ptr, int requiredLane, unsigned int * load, int warpSize, int numberOfTrialsBenchmark) {

    int pos = (blockIdx.x * warpSize) + threadIdx.x;
    int mulp;
    int warp;
    int lane;
    asm volatile ("mov.u32 %0, %%smid;" : "=r"(mulp));
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warp));
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane));
    if ((lane == requiredLane) && ((warp == 0) || (warp == 1) || (warp == 2) || (warp == 3))) {
        (*ptr).mulp[pos] = mulp;
        (*ptr).warp[pos] = warp;
        (*ptr).lane[pos] = lane;
        long long int startTime;
        long long int endTime;
        unsigned int value;
        value = 0;
        for (int preparationLoop = 0; preparationLoop < numberOfTrialsBenchmark; preparationLoop++) {
            asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(value) : "l"(load) : "memory");
            asm volatile ("add.u32 %0, %1, %2;" : "=r"(value) : "r"(value), "r"(2));
        }
        value = 0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime));
        for (int measureLoop = 0; measureLoop < numberOfTrialsBenchmark; measureLoop++) {
            asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(value) : "l"(load) : "memory");
            asm volatile ("add.u32 %0, %1, %2;" : "=r"(value) : "r"(value), "r"(2));
        }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime));
        (*ptr).time[pos] = ((float) (endTime - startTime)) / ((float) numberOfTrialsBenchmark);
        printf("%.12f ", (*ptr).time[pos]);
    }
}

/**
 * Launches the L1 benchmarks with small data collection.
 * @param ptr The small data collection where the data of the benchmarks is stored.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 */
void launchSmallL1Benchmarks(SmallDataCollection *ptr, GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    for (int laneLoop = 0; laneLoop < info.warpSize; laneLoop++) {
        unsigned int *hostLoad;
        hostLoad = (unsigned int *) malloc(sizeof(unsigned int) * prop.numberOfTrialsBenchmark);
        unsigned int *deviceLoad;
        cudaMalloc(&deviceLoad, (sizeof(unsigned int) * prop.numberOfTrialsBenchmark));
        for (int initializeLoop = 0; initializeLoop < prop.numberOfTrialsBenchmark; initializeLoop++) {
            hostLoad[initializeLoop] = initializeLoop;
        }
        cudaMemcpy(deviceLoad, hostLoad, (sizeof(unsigned int) * prop.numberOfTrialsBenchmark), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        smallL1Benchmark<<<derivatives.smallNumberOfBlocks, info.warpSize>>>(ptr, laneLoop, deviceLoad, info.warpSize, prop.numberOfTrialsBenchmark);
        cudaDeviceSynchronize();
        cudaFree(deviceLoad);
        free(hostLoad);
    }
}

#endif //GCPVE_21_L1_CACHE_BENCHMARK_CUH

//FINISHED

