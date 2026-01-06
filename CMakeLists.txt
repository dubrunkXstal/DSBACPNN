cmake_minimum_required(VERSION 3.31.5)
project(NeuralNetwork VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# configure_file(${CMAKE_SOURCE_DIR}/... ${CMAKE_BINARY_DIR}/... COPYONLY)

find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(NeuralNetwork main.cpp)
target_link_libraries(NeuralNetwork PRIVATE Eigen3::Eigen)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
