cmake_minimum_required(VERSION 2.8)
project( vo )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package( OpenCV 3 REQUIRED )
SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )

find_package(cvsba REQUIRED )

find_package( Boost 1.58 COMPONENTS program_options REQUIRED COMPONENTS system filesystem)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package(gflags REQUIRED)
find_package(glog REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/visual_odometry/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/kalman_filter/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/sfm/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/utils/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/matcher/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/kitti/*.cpp)

LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/sfm/pba/bin)

if(OpenCV_CUDA_VERSION)
    file(GLOB SRC_FILES_FEATURES ${PROJECT_SOURCE_DIR}/src/features/cuda/*.cpp)
else()
    file(GLOB SRC_FILES_FEATURES ${PROJECT_SOURCE_DIR}/src/features/*.cpp)
endif()

add_executable( vo ${SRC_FILES} ${SRC_FILES_FEATURES})

target_link_libraries( vo ${OpenCV_LIBS} ${cvsba_LIBS} ${Boost_LIBRARIES} gflags glog pba)

# Setup testing
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

file(GLOB TEST_FILES ${PROJECT_SOURCE_DIR}/src/sfm/test/*.cpp
                     ${PROJECT_SOURCE_DIR}/src/sfm/triangulation.cpp
                     ${PROJECT_SOURCE_DIR}/src/features/*.cpp
                     ${PROJECT_SOURCE_DIR}/src/matcher/*.cpp
                     ${PROJECT_SOURCE_DIR}/src/utils/*.cpp)

# Add test cpp file
add_executable(unit_tests ${TEST_FILES})

target_link_libraries(unit_tests ${OpenCV_LIBS} ${Boost_LIBRARIES} gflags glog pba gtest gtest_main)
add_test( unit_tests unit_tests )
