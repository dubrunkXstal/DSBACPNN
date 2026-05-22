cmake_minimum_required(VERSION 3.31.5)
project(NeuralNetwork VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

add_executable(NeuralNetwork main.cpp BlackBox_impl.cpp ActivationFunctions_impl.cpp)

# target_compile_options(NeuralNetwork PRIVATE -std=c++20 -Wall -Wextra -Werror)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
