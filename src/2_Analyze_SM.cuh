#ifndef GCPVE_C_C_2_ANALYZE_SM_CUH
#define GCPVE_C_C_2_ANALYZE_SM_CUH

#include "1_Gpu_Information.cuh"

typedef struct SmSimpleAddBenchmark16bit {
    bool correctSm[65536];
    float finalTime[65536];
    float laneFinal[65536];
    float warpFinal[65536];
    float smFinal[65536];
} SmSimpleAddBenchmark16bit;

__global__ void performSmSimpleAddBenchmark(int requiredSm, int blockSize, int summandSize, SmSimpleAddBenchmark16bit *host) {
    unsigned int numberOfIterations = 2048;
    unsigned int currentSm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(currentSm));
    int pos = threadIdx.x + (blockIdx.x * blockSize);
    (*host).correctSm[pos] = false;
    if (currentSm == requiredSm) {
        unsigned int startTime;
        unsigned int endTime;
        unsigned int sumTime;
        unsigned int summand1 = summandSize;
        unsigned int summand2 = summandSize;
        unsigned int sum;
        unsigned int laneId;
        unsigned int warpId;
        unsigned int smId;
        unsigned int laneSum;
        unsigned int warpSum;
        unsigned int smSum;
        for (int i = 0; i < numberOfIterations; i++) {
            asm volatile ("mov.u32 %0, %%clock;\n"
                          "add.u32 %1 %2 %3;\n"
                          "mov.u32 %4, %%clock;\n"
                          "mov.u32 %5, %%laneid;"
                          "mov.u32 %6, %%warpid;"
                          "mov.u32 %7, %%smid;"
                          : "=r"(startTime), "=r"(sum) : "r"(summand1), "r"(summand2)
                          : "=r"(endTime), "=r"(laneId), "=r"(warpId), "=r"(smId));
            sumTime = sumTime + (endTime - startTime);
            laneSum = laneSum + laneId;
            warpSum = warpSum + warpId;
            smSum = smSum + smId;
        }
        (*host).finalTime[pos] = ((float) sumTime) / ((float) numberOfIterations);
        (*host).laneFinal[pos] = ((float) laneSum) / ((float) numberOfIterations);
        (*host).warpFinal[pos] = ((float) warpSum) / ((float) numberOfIterations);
        (*host).smFinal[pos] = ((float) smSum) / ((float) numberOfIterations);
        (*host).correctSm[pos] = true;
    }
}

SmSimpleAddBenchmark16bit analyzeSm16bit(int sm, int summandSize, GpuInformation gpuInfo) {
    int numberOfTrials = gpuInfo.multiProcessorCount * 32;
    SmSimpleAddBenchmark16bit smSimpleAddBenchmark;
    SmSimpleAddBenchmark16bit *ptr;
    ptr = &smSimpleAddBenchmark;
    cudaMallocManaged(&ptr, ((65536 + (4 * (65536 * 32))) / 8));
    if ((numberOfTrials * gpuInfo.warpSize) > 65536) {
        for (int i = 0; i < 65536; i++) {
            smSimpleAddBenchmark.correctSm[i] = false;
        }
        return smSimpleAddBenchmark;
    }
    performSmSimpleAddBenchmark<<<numberOfTrials, gpuInfo.warpSize>>>(sm, gpuInfo.warpSize, summandSize, ptr);
    cudaDeviceSynchronize();
    for (int i = 0; i < (numberOfTrials * gpuInfo.warpSize); i++) {
        smSimpleAddBenchmark.finalTime[i] = (*ptr).finalTime[i];
        smSimpleAddBenchmark.laneFinal[i] = (*ptr).laneFinal[i];
        smSimpleAddBenchmark.warpFinal[i] = (*ptr).warpFinal[i];
        smSimpleAddBenchmark.smFinal[i] = (*ptr).smFinal[i];
        smSimpleAddBenchmark.correctSm[i] = (*ptr).correctSm[i];
    }
    for (int i = (numberOfTrials * gpuInfo.warpSize); i < 65536; i++) {
        smSimpleAddBenchmark.correctSm[i] = false;
    }
    cudaFree(ptr);
    return smSimpleAddBenchmark;
}

#endif //GCPVE_C_C_2_ANALYZE_SM_CUH


/*
void createSm16bitFile(SmSimpleAddBenchmark16bit smSimpleAddBenchmark) {
    char output1[] = "Benchmark_16bit.csv";
    FILE *csv1 = fopen(output1, "w");
    for (int j = 0; j < 30; j++) {
        time = 0;
        sum = 0;
        counter = 0;
        for (int i = 0; i < 65536; i++) {
            if ((*ptr1).thread[i].smId == j) {
                sum = sum + ((double) ((*ptr1).thread[i].end - (*ptr1).thread[i].begin));
                counter = counter + 1.0;
            }
        }
        time = sum / counter;
        fprintf(csv1, "%d ; ", j);
        fprintf(csv1, "%lf \n", time);
    }
    fclose(csv1);
}

    //First benchmark
    Benchmark benchmark1;
    Benchmark *ptr1;
    ptr1 = &benchmark1;
    cudaMallocManaged(&ptr1, 15728640);
    simpleAdd<<<2048, 32>>>(16777216, ptr1);
    cudaDeviceSynchronize();
    char output1[] = "Benchmark_1.csv";
    FILE *csv1 = fopen(output1, "w");
    //printf(csv1, "smId ; averageComputationTime\n");
    float time;
    float sum;
    float counter;
    for (int j = 0; j < 30; j++) {
        time = 0;
        sum = 0;
        counter = 0;
        for (int i = 0; i < 65536; i++) {
            if ((*ptr1).thread[i].smId == j) {
                sum = sum + ((double) ((*ptr1).thread[i].end - (*ptr1).thread[i].begin));
                counter = counter + 1.0;
            }
        }
        time = sum / counter;
        fprintf(csv1, "%d ; ", j);
        fprintf(csv1, "%lf \n", time);
    }
    fclose(csv1);
    cudaFree(ptr1);


    //Second benchmark
    float time;
    float sum;
    float counter;
    char output2[] = "Benchmark_2.csv";
    FILE *csv2 = fopen(output2, "w");
    //printf(csv2, "size ; averageComputationTime\n");
    Benchmark benchmark1;
    Benchmark *ptr1;
    ptr1 = &benchmark1;
    cudaMallocManaged(&ptr1, 245760);
    //simpleAdd<<<32, 32>>>(16777216, ptr2);
    //cudaDeviceSynchronize();
    for (long i = 0; i < 256; i = i++) {
        time = 0;
        sum = 0;
        counter = 0;
        simpleAdd<<<32, 32>>>((i*65536), ptr1);
        cudaDeviceSynchronize();
        for (int j = 0; j < 960; j++) {
            if ((*ptr1).thread[j].smId == 0) {
                sum = sum + ((double) ((*ptr1).thread[j].end - (*ptr1).thread[j].begin));
                counter = counter + 1.0;
            }
        }
        time = sum / counter;
        fprintf(csv2, "%ld ; ", (i*65536));
        fprintf(csv2, "%lf \n", time);
    }
    cudaFree(ptr1);
    fclose(csv2);

    //Third benchmark
    float time;
    float sum;
    float counter;
    Benchmark benchmark3;
    Benchmark *ptr3;
    ptr3 = &benchmark3;
    cudaMallocManaged(&ptr3, 15728640);
    simpleAdd<<<2048, 32>>>(16777216, ptr3);
    cudaDeviceSynchronize();
    char output3[] = "Benchmark_3.csv";
    FILE *csv3 = fopen(output3, "w");
    //printf(csv3, "laneId ; averageComputationTime\n");
    for (int j = 0; j < 64; j++) {
        time = 0;
        sum = 0;
        counter = 0;
        for (int i = 0; i < 65536; i++) {
            if (((*ptr3).thread[i].smId == 0) && ((*ptr3).thread[i].warpId == j)) {
                printf("%lld ", (*ptr3).thread[i].end);
                printf("%lld\n", (*ptr3).thread[i].begin);
                sum = sum + ((double) (*ptr3).thread[i].end - (*ptr3).thread[i].begin);
                counter = counter + 1.0;
            }
        }
        time = sum / counter;
        fprintf(csv3, "%d ; ", j);
        fprintf(csv3, "%lf \n", time);
    }
    fclose(csv3);
    cudaFree(ptr3);

    typedef struct BenchmarkThread {
    int threadId;
    int blockId;
    int laneId;
    int warpId;
    int warpNum;
    int smId;
    int smNum;
    long long begin;
    long long end;
} BenchmarkThread;

typedef struct Benchmark {
    BenchmarkThread thread[65536];
} Benchmark;

static __device__ __inline__ int getLaneId() {
    int laneId;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

static __device__ __inline__ int getWarpId(){
    int warpId;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpId));
    return warpId;
}

static __device__ __inline__ int getWarpNum(){
    int warpNum;
    asm volatile("mov.u32 %0, %%nwarpid;" : "=r"(warpNum));
    return warpNum;
}

static __device__ __inline__ int getSmId() {
    int smId;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smId));
    return smId;
}

static __device__ __inline__ int getSmNum() {
    int smNum;
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(smNum));
    return smNum;
}

static __device__ __inline__ long long getCounter() {
    long long counter;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(counter));
    return counter;
}

__global__ void simpleAdd(long n, Benchmark *host) {





    int current = (blockIdx.x * blockDim.x) + threadIdx.x;
    (*host).thread[current].threadId = threadIdx.x;
    (*host).thread[current].blockId = blockIdx.x;
    (*host).thread[current].laneId = getLaneId();
    (*host).thread[current].warpId = getWarpId();
    (*host).thread[current].warpNum = getWarpNum();
    (*host).thread[current].smId = getSmId();
    (*host).thread[current].smNum = getSmNum();
    int x[16777216];
    int y[16777216];
    int z[16777216];
    for (long i = 0; i < n; i++) {
        x[i] = i;
        y[i] = (n-i)-1;
    }
    (*host).thread[current].begin = getCounter();
    for (long i = 0; i < n; i ++) {
        z[i] = x[i] + y[i];
    }
    (*host).thread[current].end = getCounter();
}
    */

