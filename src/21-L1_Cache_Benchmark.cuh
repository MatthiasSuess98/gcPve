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
 * @param load The load which will be used in the benchmark calculation.
 * @param requiredLane The lane id which is required for the thread.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 */
__global__ void smallL1Benchmark(SmallDataCollection *ptr, unsigned int * load, int requiredLane, GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

    int mulp;
    int warp;
    int lane;

    asm volatile ("mov.u32 %0, %%smid;" : "=r"(mulp));
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warp));
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane));

    bool validWarp = false;
    for (int hardwareWarpLoop = 0; hardwareWarpLoop < derivatives.hardwareWarpsPerSm; hardwareWarpLoop++) {
        if (warp == hardwareWarpLoop) {
            validWarp = true;
        }
    }

    if (validWarp && (lane == requiredLane)) {
        int pos = (((mulp * derivatives.smallNumberOfBlocksPerMulp) + blockIdx.x) * info.warpSize) + lane;

        (*ptr).mulp[pos] = mulp;
        (*ptr).warp[pos] = warp;
        (*ptr).lane[pos] = lane;

        long long int startTime;
        long long int endTime;
        long long int finalTime;

        unsigned int preValue = 0;
        unsigned int postValue = 0;
        unsigned int summand = 1;

        for (int preparationLoop = 0; preparationLoop < derivatives.smallNumberOfTrialsDivisor; preparationLoop++) {
            asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(preValue) : "l"(load) : "memory");
            asm volatile ("add.u32 %0, %1, %2;" : "=r"(postValue) : "r"(preValue), "r"(summand));
        }

        for (int mainLoop = 0; mainLoop < derivatives.smallNumberOfTrialsDivisor; mainLoop++) {
            preValue = 0;
            postValue = 0;
            for (int measureLoop = 0; measureLoop < derivatives.smallNumberOfTrialsDivisor; measureLoop++) {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime));
                asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(preValue) : "l"(load) : "memory");
                asm volatile ("add.u32 %0, %1, %2;" : "=r"(postValue) : "r"(preValue), "r"(summand));
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime));
                finalTime = finalTime + (endTime - startTime);
            }
        }

        (*ptr).time[pos] = finalTime;
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
        hostLoad = (unsigned int *) malloc(sizeof(unsigned int) * derivatives.smallNumberOfTrialsDivisor);

        unsigned int *deviceLoad;
        cudaMalloc(&deviceLoad, (sizeof(unsigned int) * derivatives.smallNumberOfTrialsDivisor));

        for (int initializeLoop = 0; initializeLoop < derivatives.smallNumberOfTrialsDivisor; initializeLoop++) {
            hostLoad[initializeLoop] = initializeLoop;
        }

        cudaMemcpy(deviceLoad, hostLoad, (sizeof(unsigned int) * derivatives.smallNumberOfTrialsDivisor), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        smallL1Benchmark<<<derivatives.smallNumberOfBlocksPerMulp, info.warpSize>>>(ptr, deviceLoad, laneLoop, info, prop, derivatives);
        cudaDeviceSynchronize();

        cudaFree(deviceLoad);
        free(hostLoad);
    }
}

#endif //GCPVE_21_L1_CACHE_BENCHMARK_CUH

//FINISHED

