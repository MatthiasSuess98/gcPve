#ifndef GCPVE_04_DATA_COLLECTION_CUH
#define GCPVE_04_DATA_COLLECTION_CUH

#include <vector>


/**
 * Data structure for data collections of the benchmark kernels.
 */
typedef struct dataCollection {
    std::vector<int> mulp;
    std::vector<int> lane;
    std::vector<float> timeL1;
    std::vector<float> timeSM;
    std::vector<float> timeL2;
    std::vector<float> timeGM;
} dataCollection;


#endif //GCPVE_04_DATA_COLLECTION_CUH

//FINISHED

