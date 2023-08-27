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
    int x[256];
    int y[256];
    int z[256];
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
    cudaMallocManaged(&ptr, 13631488);
    simpleAdd<<<256, 256>>>(256, ptr);
    cudaDeviceSynchronize();
    char output[] = "Benchmark.csv";
    FILE *csv = fopen(output, "w");
    fprintf(csv, "threadId ; blockId ; laneId ; warpId ; warpNum ; smId ; smNum ; begin ; end \n");
    for (int i = 0; i < 65536; i++) {
        fprintf(csv, "%d ; ", (*ptr).thread[i].threadId);
        fprintf(csv, "%d ; ", (*ptr).thread[i].blockId);
        fprintf(csv, "%d ; ", (*ptr).thread[i].laneId);
        fprintf(csv, "%d ; ", (*ptr).thread[i].warpId);
        fprintf(csv, "%d ; ", (*ptr).thread[i].warpNum);
        fprintf(csv, "%d ; ", (*ptr).thread[i].smId);
        fprintf(csv, "%d ; ", (*ptr).thread[i].smNum);
        fprintf(csv, "%lld ; ", (*ptr).thread[i].begin);
        fprintf(csv, "%lld \n", (*ptr).thread[i].end);
    }
    fclose(csv);
    cudaFree(ptr);
}

#endif //GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH

