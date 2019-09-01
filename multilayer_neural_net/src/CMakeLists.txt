cmake_minimum_required(VERSION 3.4.0)
project(MultilayerNeuralNetwork)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_SOURCE_DIR}/../../util/cmake")

# Required packages
find_package(GLM REQUIRED) # For matrix math.
message("${GLM_INCLUDE_DIR}")

set(SOURCES
    main.cpp
)

add_executable(mnn "${SOURCES}" ${GLM_libraries})
