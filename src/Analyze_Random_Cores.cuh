#ifndef GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH
#define GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH

typedef struct BenchmarkThread {
    int threadId;
    int blockId;
    int laneId;
    int warpId;
    int smId;
    long long begin;
    long long end;
} BenchmarkThread;

BenchmarkThread thread[65536];

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

static __device__ __inline__ int getSmId() {
    int smId;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smId));
    return smId;
}

static __device__ __inline__ long long getCounter() {
    long long counter;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(counter));
    return counter;
}

__global__ void simpleAdd(int n) {
    int current = (blockIdx.x * blockDim.x) + threadIdx.x;
    thread[current].threadId = threadIdx.x;
    thread[current].blockId = blockIdx.x;
    thread[current].laneId = getLaneId();
    thread[current].warpId = getWarpId();
    thread[current].smId = getSmId();
    int x[n];
    int y[n];
    int z[n];
    for (int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = (n-i)-1;
    }
    thread[current].begin = getCounter();
    for (int i = 0; i < n; i ++) {
        z[i] = x[i] + y[i];
    }
    thread[current].end = getCounter();
}

void performRandomCoreBenchmark() {
    simpleAdd<<<256, 256>>>(256);
    createRandomCoreBenchmarkFile();
}

void createRandomCoreBenchmarkFile() {
    char output[] = "Benchmark.csv";
    FILE *csv = fopen(output, "w");
    fprintf(csv, "threadId ; blockId ; laneId ; warpId ; smId ; begin ; end \n");
    for (int i = 0; i < 65536; i++) {
        fprintf(csv, "%d ; ", thread[i].threadId);
        fprintf(csv, "%d ; ", thread[i].blockId);
        fprintf(csv, "%d ; ", thread[i].laneId);
        fprintf(csv, "%d ; ", thread[i].warpId);
        fprintf(csv, "%d ; ", thread[i].smId);
        fprintf(csv, "%lld ; ", thread[i].begin);
        fprintf(csv, "%lld \n", thread[i].end);
    }
    fclose(csv);
}

#endif //GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH

