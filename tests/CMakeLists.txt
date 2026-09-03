# Integration tests

add_executable(
    bbx_integration_tests
    integration/test_mnist.cpp
    integration/test_xor.cpp
    integration/test_sin_regression.cpp
    integration/test_overfit.cpp
)

target_link_libraries(
    bbx_integration_tests
    PRIVATE
        bbx::bbx
        Catch2::Catch2WithMain
)

include(Catch)

catch_discover_tests(
    bbx_integration_tests
    TEST_PREFIX "integration: "
    PROPERTIES
        LABELS "integration"
)

# Unit tests

add_executable(
    bbx_unit_tests
    unit/test_activation_functions.cpp
    unit/test_loss_functions.cpp
    unit/test_initializers.cpp
    unit/test_optimizers.cpp
    unit/test_block.cpp
)

target_link_libraries(
    bbx_unit_tests
    PRIVATE
        bbx::bbx
        Catch2::Catch2WithMain
)

catch_discover_tests(
    bbx_unit_tests
    TEST_PREFIX "unit: "
    PROPERTIES
        LABELS "unit"
)
