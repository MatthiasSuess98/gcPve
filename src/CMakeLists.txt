include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(cuda-samples/Common)
link_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(gcPve 0_Main.cu)

set_property(TARGET gcPve PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(gcPve PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gcPve PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_link_options("-lcuda")
#add_link_options("-lnvidia-ml")

message(${CUDA_TOOLKIT_ROOT_DIR})

IF(UNIX)
    #target_link_libraries( gcPve ${CUDA_LIBRARIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so "stdc++fs" )
    target_link_libraries( gcPve ${CUDA_LIBRARIES} "stdc++fs" )
ENDIF()

IF(WIN32)
    target_link_libraries( gcPve "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib")
    #target_link_libraries( gcPve "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvml.lib")
ENDIF()

IF(DEFINED COMPUTE_CAP_COMPILE_OPTIONS)
    target_compile_options(gcPve ${COMPUTE_CAP_COMPILE_OPTIONS})
else()
    target_compile_options(gcPve PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()

if ("${IsDebug}" STREQUAL "1")
    add_definitions(-DIsDebug)
endif()

install(TARGETS gcPve DESTINATION bin)

