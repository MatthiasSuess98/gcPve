# gcPve -- GPU-Core Performance Variation Explorer
This project provides an application to explore if there are any performance variations of the single GPU cores of a 
specific GPU and analyze them. Please take into account, that this project only supports Nvidia GPUs.

## Pre-requisites
- cmake 3.22 or higher
- nvcc 11.0 or higher

## Compatibility
- The app was tested on the following four GPUs: NVIDIA Tesla A100, NVIDIA T1000, NVIDIA Quadro P6000, NVIDIA Tesla K20.
- The app was only tested on Linux machines, but should also work on Windows machines.

## Installation
The installation of gcPve is primary done via Cmake:
- Pull the latest version of the cuda-samples with git submodule:
```bash
git submodule update --init
```
- Create a specific directory for the benchmarking executable:
```bash
mkdir build && cd build
```
- Build the benchmarking executable:
```bash
cmake ..
```
- Install the benchmarking executable:
```bash
make all install
```

## Running the Benchmarks
Run the benchmarking executable:
```
../inst-dir/mt4g
```
Options:
- "-adv": The final file will provide advanced GPU information.
- "-fas": The benchmarks will be executed faster than normal.
When the benchmarks are finished, the final data will be stored in a csv-file named "Output".

## Known issues
- No issues so far.

## About
gcPve was created by Matthias Suess (e-mail@matthias-suess.com) as part of his bachelor's thesis. gcPve is available 
under the Apache-2.0 license (see LICENCE-file).

