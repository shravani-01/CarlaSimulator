# Top-level build for the C++ core (geometry / real-time path).
# Phase 0: skeleton only. Real targets (vo, slam, fusion, calib) get added in Phase 2+.
cmake_minimum_required(VERSION 3.18)
project(carla_perception LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Dependencies (added as the C++ core grows) ---
# find_package(Eigen3 REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(Ceres REQUIRED)      # numerical optimization backend
# find_package(pybind11 REQUIRED)   # python bindings

# --- Core library (header-only placeholder for now) ---
add_library(carla_perception_core INTERFACE)
target_include_directories(carla_perception_core
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/core_cpp/include)

# Module subdirectories are enabled phase by phase:
# add_subdirectory(core_cpp/vo)
# add_subdirectory(core_cpp/slam)
# add_subdirectory(core_cpp/fusion)
# add_subdirectory(core_cpp/calib)
# add_subdirectory(core_cpp/bindings)

message(STATUS "carla_perception: C++ core skeleton configured (Phase 0).")
