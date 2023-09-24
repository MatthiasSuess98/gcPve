#ifndef GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH
#define GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH

/**
 * Data structure for all properties of the benchmark.
 */
typedef struct BenchmarkProperties {

    // Variables.
    int numberOfTrialsPerform;
    int numberOfTrialsLaunch;
    int numberOfTrialsDivisor;
    int memoryOverlap;
    long long int maxDelta;
    int maxDontFit;
    int small;
    int medium;
    long large;
} BenchmarkProperties;


/**
 * Creates a data structure with all properties of the benchmark.
 * @return Data structure with all properties of the benchmark.
 */
BenchmarkProperties getBenchmarkProperties() {

    // Create the final data structure.
    BenchmarkProperties prop;

    // Initialize the properties and writes them into the final data structure.
    prop.numberOfTrialsPerform = 1;
    prop.numberOfTrialsLaunch = 10;
    prop.numberOfTrialsDivisor = 1024;
    prop.memoryOverlap = 2;
    prop.maxDelta = 0;
    prop.maxDontFit = 3;
    // Size of the data collections.
    // Warning: If these three variables get updated, update the variables in 04-Core_Characteristics also!
    prop.small = 65536;
    prop.medium = 16777216;
    prop.large = 4294967296;

    // Return the final data structure.
    return prop;
}


/**
 * Creates a csv file with all information of the given data structure.
 * @param prop The given data structure.
 */
void createPropFile(BenchmarkProperties prop) {

    // Creation and opening of the csv file.
    char output[] = "BenchProp.csv";
    FILE *csv = fopen(output, "w");

    // Writing all the information into the csv file.
    fprintf(csv, "numberOfTrialsPerform; \"%d\"\n", prop.numberOfTrialsPerform);
    fprintf(csv, "numberOfTrialsLaunch; \"%d\"\n", prop.numberOfTrialsLaunch);
    fprintf(csv, "numberOfTrialsDivisor; \"%d\"\n", prop.numberOfTrialsDivisor);
    fprintf(csv, "memoryOverlap; \"%d\"\n", prop.memoryOverlap);
    fprintf(csv, "maxDelta; \"%lld\"\n", prop.maxDelta);
    fprintf(csv, "maxDontFit; \"%d\"\n", prop.maxDontFit);
    fprintf(csv, "small; \"%d\"\n", prop.small);
    fprintf(csv, "medium; \"%d\"\n", prop.medium);
    fprintf(csv, "large; \"%ld\"\n", prop.large);

    // Close the csv file.
    fclose(csv);
    printf("[INFO] The benchmark properties file was created.");
}

#endif //GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH

//FINISHED

