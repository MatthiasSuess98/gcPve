#ifndef GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH
#define GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH


/**
 * Data structure for all properties of the benchmark.
 */
typedef struct BenchmarkProperties {

    // Variables.
    int collectionFactor;
    int numberOfTrialsPerform;
    int numberOfTrialsBenchmark;
    float maxDelta;
    int maxDontFit;
} BenchmarkProperties;


/**
 * Creates a data structure with all properties of the benchmark.
 * @return Data structure with all properties of the benchmark.
 */
BenchmarkProperties getBenchmarkProperties() {

    // Create the final data structure.
    BenchmarkProperties prop;

    // Initialize the properties and writes them into the final data structure.
    prop.collectionFactor = 32;
    prop.numberOfTrialsPerform = 32;
    // Warning: When you change the next variable you also have to change the numbers in 21-L1_SM_L2_GM_Benchmark.cuh!
    prop.numberOfTrialsBenchmark = 1024;
    prop.maxDelta = 0.5;
    prop.maxDontFit = 5;

    // Return the final data structure.
    return prop;
}


/**
 * Creates a csv file with all information of the given data structure.
 * @param prop The given data structure.
 */
void createPropFile(BenchmarkProperties prop) {

    // Creation and opening of the csv file.
    char output[] = "raw/Bench_Prop.csv";
    FILE *csv = fopen(output, "w");

    // Writing all the information into the csv file.
    fprintf(csv, "collectionFactor;\"%d\"\n", prop.collectionFactor);
    fprintf(csv, "numberOfTrialsPerform;\"%d\"\n", prop.numberOfTrialsPerform);
    fprintf(csv, "numberOfTrialsBenchmark;\"%d\"\n", prop.numberOfTrialsBenchmark);
    fprintf(csv, "maxDelta;\"%f\"\n", prop.maxDelta);
    fprintf(csv, "maxDontFit;\"%d\"\n", prop.maxDontFit);

    // Close the csv file.
    fclose(csv);
    printf("[INFO] The benchmark properties file was created.\n");
}


#endif //GCPVE_C_C_2_BENCHMARK_PROPERTIES_CUH

//FINISHED

