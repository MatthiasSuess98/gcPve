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
    prop.collectionFactor = 12;
    prop.numberOfTrialsPerform = 12;
    // Warning: The next variable has a limit! Choose only a value between 1 and 1024!
    prop.numberOfTrialsBenchmark = 144;
    prop.maxDelta = 1.5;
    prop.maxDontFit = 15;

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

