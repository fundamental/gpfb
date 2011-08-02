cmake_minimum_required(VERSION 2.6)
project("gpfb") #GPU-PolyPhase-Filter-Bank

#Common options
find_package(CxxTest)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -O2 --fast-math")
set(CMAKE_BUILD_TYPE DEBUG)

#Cuda Setup
find_package(CUDA QUIET REQUIRED)
find_package(CUDA REQUIRED)

#Cuda library
CUDA_ADD_LIBRARY(cuda process.cu siggen.cpp OPTIONS -use_fast_math
                                                    -code=compute_20
                                                    -arch=compute_20)
CUDA_ADD_CUFFT_TO_TARGET(cuda)
CUDA_BUILD_CLEAN_TARGET()

#RDBE library
add_library(rdbe packet.cpp rdbe.cpp)

#Main Progam
add_executable(pfb main.cpp)
target_link_libraries(pfb cuda rdbe)

#Speed Testing
add_executable(rt-summary realtime.cpp)
target_link_libraries(rt-summary cuda rdbe)

#Spectrum analysis
add_executable(spectrum-summary spectrum.cpp)
target_link_libraries(spectrum-summary cuda)

#Packet Dump
add_executable(pkt-dump pkt-dump.cpp)
target_link_libraries(pkt-dump rdbe)

set(HERE ${CMAKE_CURRENT_SOURCE_DIR})
#Code sanity verification
if(CXXTEST_FOUND)
    enable_testing()
    CXXTEST_ADD_TEST(fir-test fir-test.cpp ${HERE}/test/fir-test.h)
    CXXTEST_ADD_TEST(pfb-test pfb-test.cpp ${HERE}/test/freq-response.h)
    CXXTEST_ADD_TEST(qpfb-test qpfb-test.cpp ${HERE}/test/fq-fixed.h)
    CXXTEST_ADD_TEST(quant-test quant-test.cpp ${HERE}/test/fixed-pt.h)
    target_link_libraries(fir-test cuda)
    target_link_libraries(pfb-test cuda)
    target_link_libraries(qpfb-test cuda)
    target_link_libraries(quant-test cuda)
endif()

#CPack
install(TARGETS rt-summary spectrum-summary pfb pkt-dump DESTINATION bin)
install(FILES README LICENSE DESTINATION doc/gpfb/)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "gpfb - GPU Polyphase Filter Bank")
set(CPACK_PACKAGE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
include(CPack)

