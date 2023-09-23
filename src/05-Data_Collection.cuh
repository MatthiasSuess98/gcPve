#ifndef GCPVE_04_DATA_COLLECTION_CUH
#define GCPVE_04_DATA_COLLECTION_CUH

#include <vector>

/**
 * Data structure for data collections.
 */
typedef struct DataCollection {
    std::vector<int> mulp;
    std::vector<int> warp;
    std::vector<int> lane;
    std::vector<long long int> time;
} DataCollection;


#endif //GCPVE_04_DATA_COLLECTION_CUH

//FINISHED

