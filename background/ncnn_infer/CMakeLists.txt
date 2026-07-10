cmake_minimum_required(VERSION 3.16)
project(naartjie_ncnn_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(OPENCV_MIN_VERSION 5.0)
find_package(OpenCV ${OPENCV_MIN_VERSION} REQUIRED COMPONENTS core imgproc imgcodecs)
message(STATUS "Using OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")
find_package(ncnn REQUIRED)

add_executable(ncnn_infer main.cpp)
target_link_libraries(ncnn_infer PRIVATE ncnn ${OpenCV_LIBS})

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(ncnn_infer PRIVATE -O3 -Wall -Wextra)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(ncnn_infer PRIVATE -mcpu=cortex-a72)
  endif()
endif()
