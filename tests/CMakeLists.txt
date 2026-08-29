add_executable(
    bbx_tests
    test_mnist.cpp
)

target_link_libraries(
    bbx_tests
    PRIVATE
        bbx::bbx
        Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(bbx_tests)
