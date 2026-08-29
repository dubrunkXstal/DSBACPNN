cmake_minimum_required(VERSION 3.20)

project(
    bbx
    VERSION 0.1.0
    LANGUAGES CXX
)

include(FetchContent)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.1
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(Eigen3)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(Catch2)

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

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
