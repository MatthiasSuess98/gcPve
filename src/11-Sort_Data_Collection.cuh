#ifndef GCPVE_11_SORT_DATA_COLLECTION_CUH
#define GCPVE_11_SORT_DATA_COLLECTION_CUH

#include <vector>

#include "01-Gpu_Information.cuh"
#include "02-Benchmark_Properties.cuh"
#include "03-Info_Prop_Derivatives.cuh"
#include "04-Core_Characteristics.cuh"
#include "05-Data_Collection.cuh"
#include "20-L1_Cache_Launcher.cuh"

/**
 * Performs the small benchmark.
 * @param info All available information of the current GPU.
 * @param prop All properties of the benchmarks.
 * @param derivatives All derivatives of info and prop.
 */
std::vector<CoreCharacteristics> sortDataCollection(info, prop, derivatives, dataCollection data, std::vector<CoreCharacteristics> gpuCores) {


    for (int blockLoop = 0; blockLoop < prop.small; blockLoop = blockLoop + info.warpSize) {
        if (data.time[blockLoop] != 0) {
            for (int laneLoop = 0; laneLoop < info.warpSize; laneLoop++) {
currentTime = gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + (data.warp[blockLoop] * info.warpSize) + laneLoop].getTypicalL1Time();
gpuCores[(data.mulp[blockLoop] * derivatives.hardwareWarpsPerSm * info.warpSize) + (data.warp[blockLoop] * info.warpSize) + laneLoop].setTypicalL1Time(((((long double) data.time[blockLoop + laneLoop]) / ((long double) (derivatives.smallNumberOfTrialsDivisor * derivatives.smallNumberOfTrialsDivisor))) + currentTime) / 2);
}
}
}
}


}

#endif //GCPVE_11_SORT_DATA_COLLECTION_CUH
