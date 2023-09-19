#ifndef GCPVE_21_L1_CACHE_BENCHMARK_CUH
#define GCPVE_21_L1_CACHE_BENCHMARK_CUH

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"

/**
 *
 * @param requiredSm
 * @param blockSize
 * @param summandSize
 * @param host
 */
__global__ void smallL1Benchmark(int requiredSm, int blockSize, int summandSize, SmSimpleAddBenchmark16bit *host) {
    unsigned int numberOfIterations = 65536;
    unsigned int currentSm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(currentSm));
    int pos = threadIdx.x + (blockIdx.x * blockSize);
    (*host).correctSm[pos] = false;
    if (currentSm == requiredSm) {
        unsigned int startTime;
        unsigned int endTime;
        float  sumTime;
        unsigned int summand1 = summandSize;
        unsigned int summand2 = summandSize;
        unsigned int sum;
        unsigned int laneId;
        unsigned int warpId;
        unsigned int smId;
        float  laneSum;
        float  warpSum;
        float  smSum;
        asm volatile (".reg.u32 t1;\n\t"
                      ".reg.u32 t2;\n\t"
                      ".reg.u32 t3;");
        float sumTimes;
        float laneSums;
        float warpSums;
        float smSums;
        for (int i = 0; i < numberOfIterations; i++) {
            asm volatile ("mov.u32 t1, %6;\n\t"
                          "mov.u32 t2, %7;\n\t"
                          "mov.u32 %0, %%clock;\n\t"
                          "add.u32 t3, t1, t2;\n\t"
                          "mov.u32 %2, %%clock;\n\t"
                          "mov.u32 %1, t3;\n\t"
                          "mov.u32 %3, %%laneid;\n\t"
                          "mov.u32 %4, %%warpid;\n\t"
                          "mov.u32 %5, %%smid;"
                    : "=r"(startTime), "=r"(sum), "=r"(endTime), "=r"(laneId), "=r"(warpId), "=r"(smId)
                    : "r"(summand1), "r"(summand2));
            /*asm volatile ("mov.u32 %0, %%clock;\n\t"
                          "add.u32 %1, %3, %4;\n\t"
                          "mov.u32 %2, %%clock;\n\t": "=r"(startTime), "=r"(sum), "=r"(endTime) : "r"(summand1), "r"(summand2));
            asm volatile ("mov.u32 %0, %%laneid;\n\t"
                          "mov.u32 %1, %%warpid;\n\t"
                          "mov.u32 %2, %%smid;"
                    : "=r"(laneId), "=r"(warpId), "=r"(smId));*/
            sumTimes = (float) (endTime - startTime);
            laneSums = (float) laneId;
            warpSums = (float) warpId;
            smSums = (float) smId;
            sum = 0;
            sumTime = sumTime + sumTimes;
            laneSum = laneSum + laneSums;
            warpSum = warpSum + warpSums;
            smSum = smSum + smSums;
        }
        (*host).finalTime[pos] = sumTime / ((float) numberOfIterations);
        (*host).laneFinal[pos] = laneSum / ((float) numberOfIterations);
        (*host).warpFinal[pos] = warpSum / ((float) numberOfIterations);
        (*host).smFinal[pos] = smSum / ((float) numberOfIterations);
        (*host).correctSm[pos] = true;
    }
}

/**
 *
 */
launchSmallL1Benchmark(ptr, numberOfBlocks) {

performSmSimpleAddBenchmark<<<numberOfTrials, gpuInfo.warpSize>>>(sm, gpuInfo.warpSize, summandSize, ptr);

}

#endif //GCPVE_21_L1_CACHE_BENCHMARK_CUH

