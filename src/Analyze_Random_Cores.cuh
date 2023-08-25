//
// Created by Matthias Suess on 25.08.23.
//

#ifndef GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH
#define GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH

__global__ void add(int n, float *x, float *y)
{

    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

typedef struct BenchmarkThread {
    int threadId;
    int blockId
    int computationTime;
};

typedef struct RandomCoreBenchmark {
    char operation[256];
    int computationTime[32];
    int


} RandomCoreBenchmark;

RandomCoreBenchmark performRandomCoreBenchmark(int numberOfThreads, int numberOfThreadBlocks) {

}

void createRandomCoreBenchmarkFile(RandomCoreBenchmark benchmark) {

}

#endif //GCPVE_C_C_ANALYZE_RANDOM_CORES_CUH
