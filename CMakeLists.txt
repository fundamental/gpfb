cmake_minimum_required(VERSION 2.6)
project(GPU-PolyPhase-Filter-Bank)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
FIND_PACKAGE(CUDA REQUIRED)

#Main description
CUDA_ADD_LIBRARY(libpfb process.cu siggen.cpp)
ADD_EXECUTABLE(pfb main.cpp)
target_link_libraries(pfb libpfb)
find_package(CUDA QUIET REQUIRED)
CUDA_ADD_CUFFT_TO_TARGET(libpfb)

CUDA_BUILD_CLEAN_TARGET()

set(CMAKE_BUILD_TYPE DEBUG)

set(HERE ${CMAKE_CURRENT_SOURCE_DIR})
#Testing information
FIND_PACKAGE(CxxTest)
if(CXXTEST_FOUND)
    enable_testing()
    CXXTEST_ADD_TEST(fir-test fir-test.cpp ${HERE}/test/fir-test.h)
    CXXTEST_ADD_TEST(pfb-test pfb-test.cpp ${HERE}/test/freq-response.h)
    CXXTEST_ADD_TEST(spec-test spec-test.cpp ${HERE}/test/spectrum.h)
    CXXTEST_ADD_TEST(quant-test quant-test.cpp ${HERE}/test/fixed-pt.h)
    target_link_libraries(fir-test libpfb)
    target_link_libraries(pfb-test libpfb)
    target_link_libraries(spec-test libpfb)
    target_link_libraries(quant-test libpfb)
endif()
