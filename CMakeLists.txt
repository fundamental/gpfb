cmake_minimum_required(VERSION 2.6)
project("GPU PolyPhase Filter Bank")

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
FIND_PACKAGE(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -O2 --fast-math")

#Main description
CUDA_ADD_LIBRARY(pfb_lib process.cu siggen.cpp OPTIONS -use_fast_math
        -code=compute_20 -arch=compute_20)
ADD_EXECUTABLE(pfb main.cpp)
target_link_libraries(pfb pfb_lib rdbe_lib)
find_package(CUDA QUIET REQUIRED)
CUDA_ADD_CUFFT_TO_TARGET(pfb_lib)

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
    CXXTEST_ADD_TEST(rtest rtest.cpp ${HERE}/test/real-time.h)
    target_link_libraries(fir-test pfb_lib)
    target_link_libraries(pfb-test pfb_lib)
    target_link_libraries(spec-test pfb_lib)
    target_link_libraries(quant-test pfb_lib)
    target_link_libraries(rtest pfb_lib rdbe_lib)
endif()

add_library(rdbe_lib packet.c rdbe.c)
set(CMAKE_C_FLAGS "-g -O2 --std=gnu99 -Wall -Wextra")
add_definitions(-DVERBOSE=1)
