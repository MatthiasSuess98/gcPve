#ifndef GCPVE_C_C_1_GPU_INFORMATION_CUH
#define GCPVE_C_C_1_GPU_INFORMATION_CUH

/**
 * Data structure for all available information of the current gpu.
 */
typedef struct GpuInformation {

    // Variables.
    int gpuId;
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture2D[2];
    int maxTexture3D[3];
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
} GpuInformation;


/**
 * Function which determines all available information of the current gpu.
 * @param gpuId Id of the selected gpu.
 * @return All available information of the selected gpu.
 */
GpuInformation getGpuInformation(int gpuId) {

    // Create the final data structure.
    GpuInformation info;
    cudaDeviceProp deviceInfo{};
    cudaGetDeviceProperties(&deviceInfo, gpuId);

    // Determines all available information and write it into the final data structure.
    info.gpuId = gpuId;
    strcpy(info.name, deviceInfo.name);
    info.totalGlobalMem = deviceInfo.totalGlobalMem;
    info.sharedMemPerBlock = deviceInfo.sharedMemPerBlock;
    info.regsPerBlock = deviceInfo.regsPerBlock;
    info.warpSize = deviceInfo.warpSize;
    info.memPitch = deviceInfo.memPitch;
    info.maxThreadsPerBlock = deviceInfo.maxThreadsPerBlock;
    info.maxThreadsDim[3] = deviceInfo.maxThreadsDim[3];
    info.maxGridSize[3] = deviceInfo.maxGridSize[3];
    info.clockRate = deviceInfo.clockRate;
    info.totalConstMem = deviceInfo.totalConstMem;
    info.major = deviceInfo.major;
    info.minor = deviceInfo.minor;
    info.textureAlignment = deviceInfo.textureAlignment;
    info.deviceOverlap = deviceInfo.deviceOverlap;
    info.multiProcessorCount = deviceInfo.multiProcessorCount;
    info.kernelExecTimeoutEnabled = deviceInfo.kernelExecTimeoutEnabled;
    info.integrated = deviceInfo.integrated;
    info.canMapHostMemory = deviceInfo.canMapHostMemory;
    info.computeMode = deviceInfo.computeMode;
    info.maxTexture1D = deviceInfo.maxTexture1D;
    info.maxTexture2D[2] = deviceInfo.maxTexture2D[2];
    info.maxTexture3D[3] = deviceInfo.maxTexture3D[3];
    info.maxTexture1DLayered[2] = deviceInfo.maxTexture1DLayered[2];
    info.maxTexture2DLayered[3] = deviceInfo.maxTexture2DLayered[3];
    info.surfaceAlignment = deviceInfo.surfaceAlignment;
    info.concurrentKernels = deviceInfo.concurrentKernels;
    info.ECCEnabled = deviceInfo.ECCEnabled;
    info.pciBusID = deviceInfo.pciBusID;
    info.pciDeviceID = deviceInfo.pciDeviceID;
    info.pciDomainID = deviceInfo.pciDomainID;
    info.tccDriver = deviceInfo.tccDriver;
    info.asyncEngineCount = deviceInfo.asyncEngineCount;
    info.unifiedAddressing = deviceInfo.unifiedAddressing;
    info.memoryClockRate = deviceInfo.memoryClockRate;
    info.memoryBusWidth = deviceInfo.memoryBusWidth;
    info.l2CacheSize = deviceInfo.l2CacheSize;
    info.maxThreadsPerMultiProcessor = deviceInfo.maxThreadsPerMultiProcessor;

    //Return the final data structure.
    return info;
}


/**
 * Creates a csv file with all information of the given data structure.
 * @param info The given data structure.
 */
void createInfoFile(GpuInformation info) {

    // Creation and opening of the csv file.
    char output[] = "GPU_Info.csv";
    FILE *csv = fopen(output, "w");

    // Writing all the information into the csv file.
    fprintf(csv, "name; \"%s\"\n", info.name);
    fprintf(csv, "totalGlobalMem; \"%zu\"\n", info.totalGlobalMem);
    fprintf(csv, "sharedMemPerBlock; \"%zu\"\n", info.sharedMemPerBlock);
    fprintf(csv, "regsPerBlock; \"%d\"\n", info.regsPerBlock);
    fprintf(csv, "warpSize; \"%d\"\n", info.warpSize);
    fprintf(csv, "memPitch; \"%zu\"\n", info.memPitch);
    fprintf(csv, "maxThreadsPerBlock; \"%d\"\n", info.maxThreadsPerBlock);
    fprintf(csv, "maxThreadsDim[3]; \"%d\"\n", info.maxThreadsDim[3]);
    fprintf(csv, "maxGridSize[3]; \"%d\"\n", info.maxGridSize[3]);
    fprintf(csv, "clockRate; \"%d\"\n", info.clockRate);
    fprintf(csv, "totalConstMem; \"%zu\"\n", info.totalConstMem);
    fprintf(csv, "major; \"%d\"\n", info.major);
    fprintf(csv, "minor; \"%d\"\n", info.minor);
    fprintf(csv, "textureAlignment; \"%zu\"\n", info.textureAlignment);
    fprintf(csv, "deviceOverlap; \"%d\"\n", info.deviceOverlap);
    fprintf(csv, "multiProcessorCount; \"%d\"\n", info.multiProcessorCount);
    fprintf(csv, "kernelExecTimeoutEnabled; \"%d\"\n", info.kernelExecTimeoutEnabled);
    fprintf(csv, "integrated; \"%d\"\n", info.integrated);
    fprintf(csv, "canMapHostMemory; \"%d\"\n", info.canMapHostMemory);
    fprintf(csv, "computeMode; \"%d\"\n", info.computeMode);
    fprintf(csv, "maxTexture1D; \"%d\"\n", info.maxTexture1D);
    fprintf(csv, "maxTexture2D[2]; \"%d\"\n", info.maxTexture2D[2]);
    fprintf(csv, "maxTexture3D[3]; \"%d\"\n", info.maxTexture3D[3]);
    fprintf(csv, "maxTexture1DLayered[2]; \"%d\"\n", info.maxTexture1DLayered[2]);
    fprintf(csv, "maxTexture2DLayered[3]; \"%d\"\n", info.maxTexture2DLayered[3]);
    fprintf(csv, "surfaceAlignment; \"%zu\"\n", info.surfaceAlignment);
    fprintf(csv, "concurrentKernels; \"%d\"\n", info.concurrentKernels);
    fprintf(csv, "ECCEnabled; \"%d\"\n", info.ECCEnabled);
    fprintf(csv, "pciBusID; \"%d\"\n", info.pciBusID);
    fprintf(csv, "pciDeviceID; \"%d\"\n", info.pciDeviceID);
    fprintf(csv, "pciDomainID; \"%d\"\n", info.pciDomainID);
    fprintf(csv, "tccDriver; \"%d\"\n", info.tccDriver);
    fprintf(csv, "asyncEngineCount; \"%d\"\n", info.asyncEngineCount);
    fprintf(csv, "unifiedAddressing; \"%d\"\n", info.unifiedAddressing);
    fprintf(csv, "memoryClockRate; \"%d\"\n", info.memoryClockRate);
    fprintf(csv, "memoryBusWidth; \"%d\"\n", info.memoryBusWidth);
    fprintf(csv, "l2CacheSize; \"%d\"\n", info.l2CacheSize);
    fprintf(csv, "maxThreadsPerMultiProcessor; \"%d\"\n", info.maxThreadsPerMultiProcessor);

    // Close the csv file.
    fclose(csv);
}

#endif //GCPVE_C_C_1_GPU_INFORMATION_CUH

//FINISHED

