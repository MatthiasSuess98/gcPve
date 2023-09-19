#ifndef GCPVE_ANALYZE_SCHEDULING_CUH
#define GCPVE_ANALYZE_SCHEDULING_CUH

typedef struct BenchmarkProperties {
    // Update in data collection
    small 65536
    medium 16777216
    large 4294967296
} BenchmarkProperties;

BenchmarkProperties getBenchmarkProperties() {
    GpuInformation info;
    cudaDeviceProp deviceInfo{};
    cudaGetDeviceProperties(&deviceInfo, gpuId);
    strcpy(info.name, deviceInfo.name);
    info.totalGlobalMem = deviceInfo.totalGlobalMem;
    info.maxNumberOfWarpsPerSm = info.maxThreadsPerMultiProcessor / info.warpSize;
    return info;
}

void createInfoFile(GpuInformation info) {
    char output[] = "GPU_Info.csv";
    FILE *csv = fopen(output, "w");
    fprintf(csv, "name; \"%s\"\n", info.name);
    fclose(csv);
}

#endif //GCPVE_ANALYZE_SCHEDULING_CUH

