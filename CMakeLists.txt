cmake_minimum_required(VERSION 3.5)
project(ExtendedKF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EKF_CORE_SOURCES
    src/tools.cpp
    src/FusionEKF.cpp
    src/kalman_filter.cpp)

add_library(ekf_core ${EKF_CORE_SOURCES})
target_include_directories(ekf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(CTest)
if(BUILD_TESTING)
  add_executable(ekf_tests tests/ekf_tests.cpp)
  target_link_libraries(ekf_tests ekf_core)
  add_test(NAME ekf_tests COMMAND ekf_tests)
endif()

option(BUILD_SIMULATOR "Build the uWebSockets simulator executable" OFF)
if(BUILD_SIMULATOR)
  add_executable(ExtendedKF src/main.cpp)
  target_link_libraries(ExtendedKF ekf_core z ssl uv uWS)
endif()
