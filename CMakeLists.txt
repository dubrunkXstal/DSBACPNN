cmake_minimum_required(VERSION 3.20)

project(
    bbx
    VERSION 0.1.0
    LANGUAGES CXX
)

include(FetchContent)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(Eigen3)

add_library(bbx INTERFACE)
add_library(bbx::bbx ALIAS bbx)

target_include_directories(
    bbx
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(
    bbx
    INTERFACE
        cxx_std_20
)

target_link_libraries(
    bbx
    INTERFACE
        Eigen3::Eigen
)
