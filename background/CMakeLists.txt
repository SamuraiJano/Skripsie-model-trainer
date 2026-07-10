cmake_minimum_required(VERSION 3.20)
project(naartjie_defect_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FETCH_BACKEND_DEPS "Fetch cpp-httplib and nlohmann/json when not installed" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime root directory")

set(OPENCV_MIN_VERSION 5.0)
if(WIN32)
  set(OpenCV_STATIC OFF CACHE BOOL "Use OpenCV shared libraries" FORCE)
endif()
find_package(OpenCV ${OPENCV_MIN_VERSION} REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
message(STATUS "Using OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to your ONNX Runtime installation directory.")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include")
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64")

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "Could not find ONNX Runtime headers/library under ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
endif()

find_package(httplib CONFIG QUIET)
find_package(nlohmann_json CONFIG QUIET)

if(FETCH_BACKEND_DEPS AND (NOT httplib_FOUND OR NOT nlohmann_json_FOUND))
  include(FetchContent)
  if(NOT httplib_FOUND)
    FetchContent_Declare(
      httplib
      URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.15.3.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(httplib)
  endif()
  if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(
      nlohmann_json
      URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
  endif()
endif()

add_executable(naartjie_backend
  src/main.cpp
  src/CameraManager.cpp
  src/Preprocessor.cpp
  src/YoloSegONNX.cpp
  src/ApiServer.cpp
  src/ResultLogger.cpp
)

target_include_directories(naartjie_backend PRIVATE
  src
  ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(naartjie_backend PRIVATE
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIBRARY}
  httplib::httplib
  nlohmann_json::nlohmann_json
)

if(WIN32)
  if(OpenCV_LIB_PATH)
    get_filename_component(OPENCV_RUNTIME_BIN_DIR "${OpenCV_LIB_PATH}/../bin" ABSOLUTE)
    file(GLOB OPENCV_RUNTIME_DLLS CONFIGURE_DEPENDS
      "${OPENCV_RUNTIME_BIN_DIR}/opencv_world*.dll"
      "${OPENCV_RUNTIME_BIN_DIR}/opencv_videoio_ffmpeg*.dll"
      "${OPENCV_RUNTIME_BIN_DIR}/opencv_videoio_msmf*.dll"
    )
    foreach(OPENCV_RUNTIME_DLL IN LISTS OPENCV_RUNTIME_DLLS)
      if(NOT OPENCV_RUNTIME_DLL MATCHES "d\\.dll$")
        add_custom_command(TARGET naartjie_backend POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_RUNTIME_DLL}"
            "$<TARGET_FILE_DIR:naartjie_backend>"
        )
      endif()
    endforeach()
  endif()

  foreach(ONNXRUNTIME_DLL_NAME onnxruntime.dll onnxruntime_providers_shared.dll)
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/${ONNXRUNTIME_DLL_NAME}")
    if(EXISTS "${ONNXRUNTIME_DLL}")
      add_custom_command(TARGET naartjie_backend POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${ONNXRUNTIME_DLL}"
          "$<TARGET_FILE_DIR:naartjie_backend>"
      )
    endif()
  endforeach()
endif()

if(MSVC)
  target_compile_options(naartjie_backend PRIVATE /W4)
else()
  target_compile_options(naartjie_backend PRIVATE -O3 -Wall -Wextra -pedantic)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(naartjie_backend PRIVATE -mcpu=cortex-a72)
  endif()
endif()
