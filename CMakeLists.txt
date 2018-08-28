cmake_minimum_required(VERSION 2.8)
project( vo )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package( OpenCV 3 REQUIRED )

find_package( Boost 1.58 COMPONENTS program_options REQUIRED COMPONENTS system filesystem)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package(gflags REQUIRED)
find_package(glog REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/visual_odemetry/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/kalman_filter/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/sfm/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/utils/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/kitti/*.cpp)

if(OpenCV_CUDA_VERSION)
    file(GLOB SRC_FILES_FEATURES ${PROJECT_SOURCE_DIR}/src/features/cuda/*.cpp)
else()
    file(GLOB SRC_FILES_FEATURES ${PROJECT_SOURCE_DIR}/src/features/*.cpp)
endif()

add_executable( vo ${SRC_FILES} ${SRC_FILES_FEATURES})

target_link_libraries( vo ${OpenCV_LIBS} ${Boost_LIBRARIES} gflags glog)

