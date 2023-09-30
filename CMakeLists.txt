cmake_minimum_required(VERSION 3.0)
project(gcPve CUDA)
enable_language(CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_INSTALL_PREFIX ..)



add_subdirectory(src)

