#ifndef GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH
#define GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH

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

__global__ void simpleAdd(int n, Benchmark *host) {
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
    for (int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = (n-i)-1;
    }
    (*host).thread[current].begin = getCounter();
    for (int i = 0; i < n; i ++) {
        z[i] = x[i] + y[i];
    }
    (*host).thread[current].end = getCounter();
}

void performRandomCoreBenchmark() {
    Benchmark benchmark;
    Benchmark *ptr;
    ptr = &benchmark;

    //First benchmark
    cudaMallocManaged(&ptr, 872415232);
    simpleAdd<<<2048, 2048>>>(2048, ptr);
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
        for (int i = 0; i < 4194304; i++) {
            if ((*ptr).thread[i].smId == j) {
                sum = sum + (((float) (*ptr).thread[i].end) - ((float) (*ptr).thread[i].begin));
                counter = counter + 1.0;
            }
        }
        printf(sum);
        printf(counter);
        time = sum / counter;
        fprintf(csv1, "%d ; ", j);
        fprintf(csv1, "%f \n", time);
    }
    fclose(csv1);
    cudaFree(ptr);

    //Second benchmark
    char output2[] = "Benchmark_2.csv";
    FILE *csv2 = fopen(output2, "w");
    //printf(csv2, "size ; averageComputationTime\n");
    for (long long i = 0; i < 4294967296; i = i + 65536) {
        time = 0;
        sum = 0;
        counter = 0;
        cudaMallocManaged(&ptr, 199680);
        simpleAdd<<<30, 32>>>(i, ptr);
        cudaDeviceSynchronize();
        for (int j = 0; j < 960; j++) {
            if ((*ptr).thread[j].smId == 0) {
                sum = sum + (((float) (*ptr).thread[j].end) - ((float) (*ptr).thread[j].begin));
                counter = counter + 1.0;
            }
        }
        time = sum / counter;
        fprintf(csv2, "%lld ; ", i);
        fprintf(csv2, "%f \n", time);
    }
    fclose(csv2);
    cudaFree(ptr);

    //Third benchmark
    cudaMallocManaged(&ptr, 872415232);
    simpleAdd<<<2048, 2048>>>(2048, ptr);
    cudaDeviceSynchronize();
    char output3[] = "Benchmark_3.csv";
    FILE *csv3 = fopen(output3, "w");
    //printf(csv3, "laneId ; averageComputationTime\n");
    for (int j = 0; j < 32; j++) {
        time = 0;
        sum = 0;
        counter = 0;
        for (int i = 0; i < 4194304; i++) {
            if (((*ptr).thread[i].smId == 0) && ((*ptr).thread[i].laneId == j)) {
                sum = sum + (((float) (*ptr).thread[i].end) - ((float) (*ptr).thread[i].begin));
                counter = counter + 1.0;
            }
        }
        time = sum / counter;
        fprintf(csv3, "%d ; ", j);
        fprintf(csv3, "%f \n", time);
    }
    fclose(csv3);
    cudaFree(ptr);
}

#endif //GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH

