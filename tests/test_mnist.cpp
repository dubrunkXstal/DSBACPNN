#include <catch2/catch_test_macros.hpp>

#include <bbx/blackbox/BlackBox.hpp>

// Датасет: https://github.com/phoebetronic/mnist

TEST_CASE("MNIST", "[integration]")
{
    bbx::BlackBox bb{
        {784, 500, bbx::Sigmoid{}},
        {500, 100, bbx::Sigmoid{}},
        {100, 100, bbx::Relu{}},
        {100, 200, bbx::Sigmoid{}},
        {200, 100, bbx::Sigmoid{}},
        {100, 10, bbx::Sigmoid{}}
    };

    REQUIRE(bb.getBlocksCount() == 6);

    bb.loadTrainCSV(
        "",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    );

    double score = 0.0;
    score = bb.loadTestCSV(
        "",
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        [](const bbx::Vector& x, const bbx::Vector& y) {
            bbx::Index x_argmax, y_argmax;
            x.maxCoeff(&x_argmax);
            y.maxCoeff(&y_argmax);
            return x_argmax == y_argmax;
    });

    CAPTURE(score);
    FAIL("Forcing failure to view captures");
}
