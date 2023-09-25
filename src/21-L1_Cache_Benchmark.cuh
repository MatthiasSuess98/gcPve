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
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 */
__global__ void smallL1Benchmark(SmallDataCollection *ptr, unsigned int * load, GpuInformation info, BenchmarkProperties prop, InfoPropDerivatives derivatives) {

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

    if (!validWarp) {
        (*ptr).ctrl[blockIdx.x] = true;
    } else {
        bool allTrue = true;
        unsigned int counter = 0;
        while (counter < info.warpSize) {

            allTrue = true;
            for (int blockLoop = 0; blockLoop < derivatives.smallTotalNumberOfBlocks; blockLoop++) {
                if ((*ptr).ctrl[blockLoop] == false) {
                    allTrue = false;
                }
            }

            if (allTrue) {
                counter++;
                (*ptr).ctrl[blockIdx.x] = false;
            }

            if (((*ptr).ctrl[blockIdx.x] == false) && (couter == lane)) {

                int pos = (blockIdx.x * info.warpSize) + lane;

                (*ptr).mulp[pos] = mulp;
                (*ptr).warp[pos] = warp;
                (*ptr).lane[pos] = lane;

                // Warning: If these magical numbers get updated, update the variables in 02-Benchmark_Properties and in 04-Core_Characteristics also!
                long long int startTime[65536 / 1024];
                long long int endTime[65536 / 1024];
                long long int finalTime[65536 / 1024];

                long long int returnTime = 0;

                unsigned int preValue = 0;
                unsigned int postValue = 0;
                unsigned int summand = 1;

                for (int preparationLoop = 0; preparationLoop < derivatives.smallNumberOfTrialsDivisor; preparationLoop++) {
                    asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(preValue) : "l"(load) : "memory");
                    asm volatile ("add.u32 %0, %1, %2;" : "=r"(postValue) : "r"(preValue), "r"(summand));
                }

                for (int resetLoop = 0; resetLoop < derivatives.smallNumberOfTrialsDivisor; resetLoop++) {
                    finalTime[resetLoop] = 0;
                }

                for (int mainLoop = 0; mainLoop < derivatives.smallNumberOfTrialsDivisor; mainLoop++) {

                    preValue = 0;
                    postValue = 0;

                    for (int resetLoop = 0; resetLoop < derivatives.smallNumberOfTrialsDivisor; resetLoop++) {
                        endTime[resetLoop] = 0;
                        startTime[resetLoop] = 0;
                    }

                    for (int measureLoop = 0; measureLoop < derivatives.smallNumberOfTrialsDivisor; measureLoop++) {
                        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime[measureLoop]));
                        asm volatile ("ld.global.ca.u32 %0, [%1];" : "=r"(preValue) : "l"(load) : "memory");
                        asm volatile ("add.u32 %0, %1, %2;" : "=r"(postValue) : "r"(preValue), "r"(summand));
                        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(endTime[measureLoop]));
                    }

                    for (int addLoop = 0; addLoop < derivatives.smallNumberOfTrialsDivisor; addLoop++) {
                        finalTime[mainLoop] = finalTime[mainLoop] + (endTime[addLoop] - startTime[addLoop]);
                    }
                }

                for (int addLoop = 0; addLoop < derivatives.smallNumberOfTrialsDivisor; addLoop++) {
                    returnTime = returnTime + finalTime[addLoop];
                }

                postValue++;
                (*ptr).time[pos] = returnTime;
                (*ptr).ctrl[blockIdx.x] = true;
            } else {
                (*ptr).ctrl[blockIdx.x] = true;
            }
        }
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

    unsigned int *hostLoad;
    hostLoad = (unsigned int *) malloc(sizeof(unsigned int) * derivatives.smallNumberOfTrialsDivisor);

    unsigned int *deviceLoad;
    cudaMalloc(&deviceLoad, (sizeof(unsigned int) * derivatives.smallNumberOfTrialsDivisor));

    for (int initializeLoop = 0; initializeLoop < derivatives.smallNumberOfTrialsDivisor; initializeLoop++) {
        hostLoad[initializeLoop] = initializeLoop;
    }

    cudaMemcpy(deviceLoad, hostLoad, (sizeof(unsigned int) * derivatives.smallNumberOfTrialsDivisor), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    //unsigned int switchValue = 0;
    //unsigned int *switchValuePtr;
    //switchValuePtr = &switchValue;
    //cudaMallocManaged(&switchValuePtr, sizeof(unsigned int));
    //cudaDeviceSynchronize();
    //@param switchValue The current lane id which is required for the current benchmark.

    smallL1Benchmark<<<derivatives.smallTotalNumberOfBlocks, info.warpSize>>>(ptr, deviceLoad, info, prop, derivatives);
    cudaDeviceSynchronize();

    cudaFree(deviceLoad);
    free(hostLoad);
    cudaFree(switchValuePtr);
}

#endif //GCPVE_21_L1_CACHE_BENCHMARK_CUH

//FINISHED

