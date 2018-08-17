cmake_minimum_required(VERSION 2.8)
project( VisualOdemetry )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package( OpenCV 3.4.2 REQUIRED )

find_package( Boost 1.58 COMPONENTS program_options REQUIRED COMPONENTS system filesystem)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package(gflags REQUIRED)
find_package(glog REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable( VisualOdemetry ${SRC_FILES})

target_link_libraries( VisualOdemetry ${OpenCV_LIBS} ${Boost_LIBRARIES} gflags glog)