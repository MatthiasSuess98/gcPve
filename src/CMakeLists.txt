include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(cuda-samples/Common)
link_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(gcPve 00-Main.cu)

set_property(TARGET gcPve PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(gcPve PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gcPve PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

install(TARGETS gcPve DESTINATION bin)

