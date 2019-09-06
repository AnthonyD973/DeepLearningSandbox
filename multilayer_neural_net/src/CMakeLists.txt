cmake_minimum_required(VERSION 3.4.0)
project(MultilayerNeuralNetwork)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# Required packages
#
set(EXTERNAL_REPOS_DIR "${CMAKE_SOURCE_DIR}/../../subprojects")
include_directories("${EXTERNAL_REPOS_DIR}/eigen")

set(SOURCES
    main.cpp
)

add_executable(mnn "${SOURCES}")
