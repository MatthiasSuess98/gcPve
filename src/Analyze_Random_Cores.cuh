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

__global__ void simpleAdd(int n, Benchmark *host) {
    int current = (blockIdx.x * blockDim.x) + threadIdx.x;
    host.thread[current].threadId = threadIdx.x;
    host.thread[current].blockId = blockIdx.x;
    host.thread[current].laneId = getLaneId();
    host.thread[current].warpId = getWarpId();
    host.thread[current].smId = getSmId();
    int x[];
    int y[];
    int z[];
    for (int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = (n-i)-1;
    }
    host.thread[current].begin = getCounter();
    for (int i = 0; i < n; i ++) {
        z[i] = x[i] + y[i];
    }
    host.thread[current].end = getCounter();
}

void performRandomCoreBenchmark() {
    Benchmark* benchmark;
    cudaMallocManaged(&benchmark, 13631488);
    simpleAdd<<<256, 256>>>(256, benchmark);
    cudaDeviceSynchronize();
    char output[] = "Benchmark.csv";
    FILE *csv = fopen(output, "w");
    fprintf(csv, "threadId ; blockId ; laneId ; warpId ; smId ; begin ; end \n");
    for (int i = 0; i < 65536; i++) {
        fprintf(csv, "%d ; ", benchmark.thread[i].threadId);
        fprintf(csv, "%d ; ", benchmark.thread[i].blockId);
        fprintf(csv, "%d ; ", benchmark.thread[i].laneId);
        fprintf(csv, "%d ; ", benchmark.thread[i].warpId);
        fprintf(csv, "%d ; ", benchmark.thread[i].smId);
        fprintf(csv, "%lld ; ", benchmark.thread[i].begin);
        fprintf(csv, "%lld \n", benchmark.thread[i].end);
    }
    fclose(csv);
    cudaFree(benchmark);
}

#endif //GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH

