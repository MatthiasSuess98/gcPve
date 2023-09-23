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

/**
 * Data structure for data collections.
 */
__device__ typedef struct DeviceCollection {
    std::vector<int> mulp;
    std::vector<int> warp;
    std::vector<int> lane;
    std::vector<long long int> time;
} DeviceCollection;

#endif //GCPVE_04_DATA_COLLECTION_CUH

//FINISHED

