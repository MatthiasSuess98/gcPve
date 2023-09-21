#ifndef GCPVE_03_INFO_PROP_DERIVATIVES_CUH
#define GCPVE_03_INFO_PROP_DERIVATIVES_CUH

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"

/**
 * Data structure for all derivatives.
 */
typedef struct InfoPropDerivatives {

    // Variables.
    float cudaVersion;
    int maxNumberOfWarpsPerSm;
    int numberOfCoresPerSm;
    int totalNumberOfCores;
    int smallNumberOfBlocks;
    int smallNumberOfBlocksPerMulp;
    int smallTotalNumberOfBlocks;
    int mediumNumberOfBlocks;
    int mediumNumberOfBlocksPerMulp;
    int mediumTotalNumberOfBlocks;
    int largeNumberOfBlocks;
    int largeNumberOfBlocksPerMulp;
    int largeTotalNumberOfBlocks;
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
    } else if ((info.ajor == 9) && (info.minor == 0)) {
        derivatives.numberOfCoresPerSm = 128;
    } else {
        derivatives.numberOfCoresPerSm = 0;
    }
    derivatives.totalNumberOfCores = derivatives.numberOfCoresPerSm * info.multiProcessorCount;
    derivatives.smallNumberOfBlocks = prop.small / info.warpSize;
    derivatives.smallNumberOfBlocksPerMulp = (prop.small / info.warpSize) / info.multiProcessorCount;
    derivatives.smallTotalNumberOfBlocks = derivatives.smallNumberOfBlocksPerMulp * info.multiProcessorCount;
    derivatives.mediumNumberOfBlocks = prop.medium / info.warpSize;
    derivatives.mediumNumberOfBlocksPerMulp = (prop.medium / info.warpSize) / info.multiProcessorCount;
    derivatives.mediumTotalNumberOfBlocks = derivatives.mediumNumberOfBlocksPerMulp * info.multiProcessorCount;
    derivatives.largeNumberOfBlocks = prop.large / info.warpSize;
    derivatives.largeNumberOfBlocksPerMulp = (prop.large / info.warpSize) / info.multiProcessorCount;
    derivatives.largeTotalNumberOfBlocks = derivatives.largeNumberOfBlocksPerMulp * info.multiProcessorCount;
    derivatives.hardwareWarpsPerSm = derivatives.numberOfCoresPerSm / info.warpSize;

    // Return the final data structure.
    return info;
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
    fprintf(csv, "smallNumberOfBlocks; \"%d\"\n", derivatives.smallNumberOfBlocks);
    fprintf(csv, "smallNumberOfBlocksPerMulp; \"%d\"\n", derivatives.smallNumberOfBlocksPerMulp);
    fprintf(csv, "mediumNumberOfBlocks; \"%d\"\n", derivatives.mediumNumberOfBlocks);
    fprintf(csv, "mediumNumberOfBlocksPerMulp; \"%d\"\n", derivatives.mediumNumberOfBlocksPerMulp);
    fprintf(csv, "largeNumberOfBlocks; \"%d\"\n", derivatives.largeNumberOfBlocks);
    fprintf(csv, "largeNumberOfBlocksPerMulp; \"%d\"\n", derivatives.largeNumberOfBlocksPerMulp);
    fprintf(csv, "hardwareWarpsPerSm; \"%d\"\n", derivatives.hardwareWarpsPerSm);

    // Close the csv file.
    fclose(csv);
}

#endif //GCPVE_03_INFO_PROP_DERIVATIVES_CUH

//FINISHED

