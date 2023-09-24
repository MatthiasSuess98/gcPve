#ifndef GCPVE_04_CORE_CHARACTERISTICS_CUH
#define GCPVE_04_CORE_CHARACTERISTICS_CUH

/**
 * Object class of the core characteristics.
 */
class CoreCharacteristics {
private:
    int smId;
    int hardwareWarpId;
    int warpCoreId;
    long long int typicalL1Time;
    long long int typicalSmTime;
    long long int typicalL2Time;
    long long int typicalGmTime;

public:
    CoreCharacteristics(int newSmId, int newHardwareWarpId, int newWarpCoreId) {
        smId = newSmId;
        hardwareWarpId = newHardwareWarpId;
        warpCoreId = newWarpCoreId;
        typicalL1Time = 0;
        typicalSmTime = 0;
        typicalL2Time = 0;
        typicalGmTime = 0;
    }

    void setTypicalL1Time(long long int newTypicalL1Time) {
        typicalL1Time = newTypicalL1Time;
    }

    void setTypicalSmTime(long long int newTypicalSmTime) {
        typicalSmTime = newTypicalSmTime;
    }

    void setTypicalL2Time(long long int newTypicalL2Time) {
        typicalL2Time = newTypicalL2Time;
    }

    void setTypicalGmTime(long long int newTypicalGmTime) {
        typicalGmTime = newTypicalGmTime;
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

    long long int getTypicalL1Time() {
        return typicalL1Time;
    }

    long long int getTypicalSmTime() {
        return typicalSmTime;
    }

    long long int getTypicalL2Time() {
        return typicalL2Time;
    }

    long long int getTypicalGmTime() {
        return typicalGmTime;
    }
};

#endif //GCPVE_04_CORE_CHARACTERISTICS_CUH

//FINISHED

