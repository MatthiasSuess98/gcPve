#ifndef GCPVE_03_INFO_PROP_DERIVATIVES_CUH
#define GCPVE_03_INFO_PROP_DERIVATIVES_CUH

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "05-Data_Collection.cuh"

/**
 * Data structure for all derivatives.
 */
typedef struct InfoPropDerivatives {

    // Variables.
    float cudaVersion;
    int maxNumberOfWarpsPerSm;
    int numberOfCoresPerSm;
    int totalNumberOfCores;
    int collectionSize;
    int numberOfBlocks;
    int numberOfBlocksPerMulp;
    int totalNumberOfBlocks;
    int hardwareWarpsPerSm;
} InfoPropDerivatives;


/**
 * Creates a data structure with all derivatives.
 * @return Data structure with all derivatives.
 */
InfoPropDerivatives getInfoPropDerivatives(GpuInformation info, BenchmarkProperties prop) {

    // Create the final data structure.
    InfoPropDerivatives derivatives;

    // Create the derivatives.
    derivatives.cudaVersion = (((float) info.major) + (((float) info.minor) / 10.));
    derivatives.maxNumberOfWarpsPerSm = info.maxThreadsPerMultiProcessor / info.warpSize;
    // Data from "helper_cuda.h".
    if ((info.major == 3) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 192;
    } else if ((info.major == 3) && (info.minor == 2)) {
        derivatives.numberOfCoresPerSm = 192;
    } else if ((info.major == 3) && (info.minor == 5)) {
        derivatives.numberOfCoresPerSm = 192;
    } else if ((info.major == 3) && (info.minor == 7)) {
        derivatives.numberOfCoresPerSm = 192;
    } else if ((info.major == 5) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 5) && (info.minor == 2)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 5) && (info.minor == 3)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 6) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 64;
    } else if ((info.major == 6) && (info.minor == 1)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 6) && (info.minor == 2)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 7) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 64;
    } else if ((info.major == 7) && (info.minor == 2)) {
        derivatives.numberOfCoresPerSm = 64;
    } else if ((info.major == 7) && (info.minor == 5)) {
        derivatives.numberOfCoresPerSm = 64;
    } else if ((info.major == 8) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 64;
    } else if ((info.major == 8) && (info.minor == 6)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 8) && (info.minor == 7)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 8) && (info.minor == 9)) {
        derivatives.numberOfCoresPerSm = 128;
    } else if ((info.major == 9) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 128;
    } else {
        derivatives.numberOfCoresPerSm = 0;
    }
    derivatives.totalNumberOfCores = derivatives.numberOfCoresPerSm * info.multiProcessorCount;
    DataCollection data;
    int iniMulp = 0;
    data.mulp.push_back(iniMulp);
    int iniWarp = 0;
    data.warp.push_back(iniWarp);
    int iniLane = 0;
    data.lane.push_back(iniLane);
    long long int iniTime = 0;
    data.time.push_back(iniTime);
    derivatives.collectionSize = info.totalGlobalMem / (sizeof(data) * prop.memoryOverlap);
    derivatives.numberOfBlocks = derivatives.collectionSize / info.warpSize;
    derivatives.numberOfBlocksPerMulp = (derivatives.collectionSize / info.warpSize) / info.multiProcessorCount;
    derivatives.totalNumberOfBlocks = derivatives.numberOfBlocksPerMulp * info.multiProcessorCount;
    derivatives.hardwareWarpsPerSm = derivatives.numberOfCoresPerSm / info.warpSize;

    // Return the final data structure.
    return derivatives;
}


/**
 * Creates a csv file with all information of the given data structure.
 * @param prop The given data structure.
 */
void createInfoPropDerivatives(InfoPropDerivatives derivatives) {

    // Creation and opening of the csv file.
    char output[] = "InfoPropDerivatives.csv";
    FILE *csv = fopen(output, "w");

    // Writing all the information into the csv file.
    fprintf(csv, "cudaVersion; \"%f\"\n", derivatives.cudaVersion);
    fprintf(csv, "maxNumberOfWarpsPerSm; \"%d\"\n", derivatives.maxNumberOfWarpsPerSm);
    fprintf(csv, "numberOfCoresPerSm; \"%d\"\n", derivatives.numberOfCoresPerSm);
    fprintf(csv, "totalNumberOfCores; \"%d\"\n", derivatives.totalNumberOfCores);
    fprintf(csv, "numberOfBlocks; \"%d\"\n", derivatives.numberOfBlocks);
    fprintf(csv, "numberOfBlocksPerMulp; \"%d\"\n", derivatives.numberOfBlocksPerMulp);
    fprintf(csv, "totalNumberOfBlocks; \"%d\"\n", derivatives.totalNumberOfBlocks);
    fprintf(csv, "collectionSize; \"%d\"\n", derivatives.collectionSize);
    fprintf(csv, "hardwareWarpsPerSm; \"%d\"\n", derivatives.hardwareWarpsPerSm);

    // Close the csv file.
    fclose(csv);
    printf("[INFO] GPU information and Benchmark properties derivatives file created.\n");
}

#endif //GCPVE_03_INFO_PROP_DERIVATIVES_CUH

//FINISHED

