# gcPve -- GPU-Core Performance Variation Explorer
This project provides an application to explore if there are any performance variations of the single GPU cores of a 
specific GPU and analyze them. Please take into account, that this project only supports Nvidia GPUs.

## Pre-requisites
- cmake 3.0 or higher
- nvcc 11.0 or higher

## Compatibility
- The app was tested on the following GPUs: NVIDIA Quadro P6000 and NVIDIA Tesla A100.
- The app was developed and tested only for Linux machines.

## Installation
The installation of gcPve is primary done via Cmake:
- Create a new directory for the gcPve files and move into it:
```bash
mkdir gcPve && cd gcPve
```
- Clone the gcPve files from the repository and move into it:
```bash
git clone https://github.com/MatthiasSuess98/gcPve && cd gcPve
```
- Create a specific directory for the installation and move into it:
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
Run the benchmarking executable in the specific directory with the following command:
```bash
cd .. && ./bin/gcPve 0
```
In this command, the GPU id of the GPU for which the benchmarks will be created is given as a parameter with the syntax above, here for GPU 0). To get a list of all available GPUs use the following command:
```bash
nvidia-smi -L
```
It is also possible to select multiple GPUs by appending multiple numbers, but at least one number must be given in total. When the benchmarks are finished, the raw benchmark results will be stored in csv-files and can be accessed with the following command:
```bash
cd raw
```
The program also creates graphical visualizations of the benchmark results which can be accessed with the following command:
```bash
cd out
```

## About
gcPve was created by Matthias Suess (e-mail@matthias-suess.com) as part of his bachelor's thesis in informatics. gcPve is available under the Apache-2.0 license (see LICENCE-file).

