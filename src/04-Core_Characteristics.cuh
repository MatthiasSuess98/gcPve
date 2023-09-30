#ifndef GCPVE_04_CORE_CHARACTERISTICS_CUH
#define GCPVE_04_CORE_CHARACTERISTICS_CUH

/**
 * Object class of the core characteristics.
 */
class CoreCharacteristics {
private:
    int numberOfCores;
    int smId;
    int hardwareWarpId;
    int warpCoreId;
    float typicalL1Time;
    float typicalSmTime;
    float typicalL2Time;
    float typicalGmTime;

public:
    CoreCharacteristics(int newNumberOfCores, int newSmId, int newHardwareWarpId, int newWarpCoreId) {
        numberOfCores = newNumberOfCores;
        smId = newSmId;
        hardwareWarpId = newHardwareWarpId;
        warpCoreId = newWarpCoreId;
        typicalL1Time = 0.0;
        typicalSmTime = 0.0;
        typicalL2Time = 0.0;
        typicalGmTime = 0.0;
    }

    void setTypicalL1Time(long double newTypicalL1Time) {
        typicalL1Time = newTypicalL1Time;
    }

    void setTypicalSmTime(long double newTypicalSmTime) {
        typicalSmTime = newTypicalSmTime;
    }

    void setTypicalL2Time(long double newTypicalL2Time) {
        typicalL2Time = newTypicalL2Time;
    }

    void setTypicalGmTime(long double newTypicalGmTime) {
        typicalGmTime = newTypicalGmTime;
    }

    int getNumberOfCores() {
        return numberOfCores;
    }

    int getSmId() {
        return smId;
    }

    int getHardwareWarpId() {
        return hardwareWarpId;
    }

    int getWarpCoreId() {
        return warpCoreId;
    }

    long double getTypicalL1Time() {
        return typicalL1Time;
    }

    long double getTypicalSmTime() {
        return typicalSmTime;
    }

    long double getTypicalL2Time() {
        return typicalL2Time;
    }

    long double getTypicalGmTime() {
        return typicalGmTime;
    }
};

#endif //GCPVE_04_CORE_CHARACTERISTICS_CUH

//FINISHED

