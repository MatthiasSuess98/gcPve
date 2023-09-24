#ifndef GCPVE_03_INFO_PROP_DERIVATIVES_CUH
#define GCPVE_03_INFO_PROP_DERIVATIVES_CUH

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"

/**
 * Data structure for all derivatives.
 */
typedef struct InfoPropDerivatives {

    // Full cuda version.
    float cudaVersion;

    // Basic derivatives.
    int maxNumberOfWarpsPerSm;
    int numberOfCoresPerSm;
    int totalNumberOfCores;
    int hardwareWarpsPerSm;

    // Small derivatives.
    int smallNumberOfBlocks;
    int smallNumberOfBlocksPerMulp;
    int smallTotalNumberOfBlocks;
    int smallNumberOfTrialsDivisor;

    // Medium derivatives.
    int mediumNumberOfBlocks;
    int mediumNumberOfBlocksPerMulp;
    int mediumTotalNumberOfBlocks;
    int mediumNumberOfTrialsDivisor;

    // Large derivatives.
    int largeNumberOfBlocks;
    int largeNumberOfBlocksPerMulp;
    int largeTotalNumberOfBlocks;
    int largeNumberOfTrialsDivisor;
} InfoPropDerivatives;


/**
 * Creates a data structure with all derivatives.
 * @return Data structure with all derivatives.
 */
InfoPropDerivatives getInfoPropDerivatives(GpuInformation info, BenchmarkProperties prop) {

    // Create the final data structure.
    InfoPropDerivatives derivatives;

    //Create full cuda version.
    derivatives.cudaVersion = (((float) info.major) + (((float) info.minor) / 10.));

    // Create basic derivatives.
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
    derivatives.hardwareWarpsPerSm = derivatives.numberOfCoresPerSm / info.warpSize;

    // Create small derivatives.
    derivatives.smallNumberOfBlocks = prop.small / info.warpSize;
    derivatives.smallNumberOfBlocksPerMulp = (prop.small / info.warpSize) / info.multiProcessorCount;
    derivatives.smallTotalNumberOfBlocks = derivatives.smallNumberOfBlocksPerMulp * info.multiProcessorCount;
    derivatives.smallNumberOfTrialsDivisor = prop.small / prop.numberOfTrialsDivisor;

    // Create medium derivatives.
    derivatives.mediumNumberOfBlocks = prop.medium / info.warpSize;
    derivatives.mediumNumberOfBlocksPerMulp = (prop.medium / info.warpSize) / info.multiProcessorCount;
    derivatives.mediumTotalNumberOfBlocks = derivatives.mediumNumberOfBlocksPerMulp * info.multiProcessorCount;
    derivatives.mediumNumberOfTrialsDivisor = prop.medium / prop.numberOfTrialsDivisor;

    // Create large derivatives.
    derivatives.largeNumberOfBlocks = prop.large / info.warpSize;
    derivatives.largeNumberOfBlocksPerMulp = (prop.large / info.warpSize) / info.multiProcessorCount;
    derivatives.largeTotalNumberOfBlocks = derivatives.largeNumberOfBlocksPerMulp * info.multiProcessorCount;
    derivatives.largeNumberOfTrialsDivisor = prop.large / prop.numberOfTrialsDivisor;

    // Return the final data structure.
    return derivatives;
}


/**
 * Creates a csv file with all information of the given data structure.
 * @param prop The given data structure.
 */
void createInfoPropDerivatives(InfoPropDerivatives derivatives) {

    // Creation and opening of the csv file.
    char output[] = "raw/InfoProp_Derivatives.csv";
    FILE *csv = fopen(output, "w");

    // Writing the full cuda version into the csv file.
    fprintf(csv, "cudaVersion; \"%f\"\n", derivatives.cudaVersion);

    // Writing the basic derivatives into the csv file.
    fprintf(csv, "maxNumberOfWarpsPerSm; \"%d\"\n", derivatives.maxNumberOfWarpsPerSm);
    fprintf(csv, "numberOfCoresPerSm; \"%d\"\n", derivatives.numberOfCoresPerSm);
    fprintf(csv, "totalNumberOfCores; \"%d\"\n", derivatives.totalNumberOfCores);
    fprintf(csv, "hardwareWarpsPerSm; \"%d\"\n", derivatives.hardwareWarpsPerSm);

    // Writing the small derivatives into the csv file.
    fprintf(csv, "smallNumberOfBlocks; \"%d\"\n", derivatives.smallNumberOfBlocks);
    fprintf(csv, "smallNumberOfBlocksPerMulp; \"%d\"\n", derivatives.smallNumberOfBlocksPerMulp);
    fprintf(csv, "smallTotalNumberOfBlocks; \"%d\"\n", derivatives.smallTotalNumberOfBlocks);
    fprintf(csv, "smallNumberOfTrialsDivisor; \"%d\"\n", derivatives.smallNumberOfTrialsDivisor);

    // Writing the medium derivatives into the csv file.
    fprintf(csv, "mediumNumberOfBlocks; \"%d\"\n", derivatives.mediumNumberOfBlocks);
    fprintf(csv, "mediumNumberOfBlocksPerMulp; \"%d\"\n", derivatives.mediumNumberOfBlocksPerMulp);
    fprintf(csv, "mediumTotalNumberOfBlocks; \"%d\"\n", derivatives.mediumTotalNumberOfBlocks);
    fprintf(csv, "mediumNumberOfTrialsDivisor; \"%d\"\n", derivatives.mediumNumberOfTrialsDivisor);

    // Writing the large derivatives into the csv file.
    fprintf(csv, "largeNumberOfBlocks; \"%d\"\n", derivatives.largeNumberOfBlocks);
    fprintf(csv, "largeNumberOfBlocksPerMulp; \"%d\"\n", derivatives.largeNumberOfBlocksPerMulp);
    fprintf(csv, "largeTotalNumberOfBlocks; \"%d\"\n", derivatives.largeTotalNumberOfBlocks);
    fprintf(csv, "largeNumberOfTrialsDivisor; \"%d\"\n", derivatives.largeNumberOfTrialsDivisor);

    // Close the csv file.
    fclose(csv);
    printf("[INFO] The info-prop-derivatives file was created.\n");
}

#endif //GCPVE_03_INFO_PROP_DERIVATIVES_CUH

//FINISHED

